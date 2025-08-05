import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

def preprocess_post(raw_file_path,processed_file_path=None):
    with open (raw_file_path,encoding='utf-8') as file:
        posts=json.load(file)
        enriched_post=[]
        for post in posts['posts']:
            metadata=extract_metadata(post['content'])
            post_with_metadata=post | metadata
            enriched_post.append(post_with_metadata)
        
        unified_tags=get_unified_tags(enriched_post)
        for post in enriched_post:
                current_tags = post['tags']
                new_tags = {unified_tags.get(tag,tag) for tag in current_tags}
                post['tags'] = list(new_tags)

        with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
            json.dump(enriched_post, outfile, indent=4)


def extract_metadata(content):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble.Only Return Json Output No extra Text. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.Tags should be only related to niche or topic not any person name.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''
    prompt=PromptTemplate.from_template(template)
    chain=prompt | llm
    response=chain.invoke(input={"post":content})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException ("Context too big. Unable to parse jobs.")
    return res

def get_unified_tags(post_with_metadata):
    unique_tags=set()

    for post in post_with_metadata:
        unique_tags.update(post['tags'])

    unique_tags_list = ','.join(unique_tags)
    template = '''I will give you a list of tags. You must output *only* a valid JSON object, and nothing else. 
Do not include any explanation, headings, or text before or after the JSON object.

Requirements:
1. Tags should be unified and merged to create a shorter list.
   Examples:
   - "Jobseekers", "Job Hunting" → "Job Search"
   - "Motivation", "Inspiration", "Drive" → "Motivation"
   - "Personal Growth", "Personal Development", "Self Improvement" → "Self Improvement"
   - "Scam Alert", "Job Scam" → "Scams"
2. All tags should use Title Case (e.g., "Self Improvement", "Job Search")
3. The output must be a **pure JSON object**, mapping original tags to unified tags.
4. Do not include markdown, text formatting, explanations, or headings. Only output the raw JSON.

Here is the list of tags:  
{tags}
'''
    prompt=PromptTemplate.from_template(template)
    chain=prompt | llm
    response=chain.invoke(input={"tags":str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException ("Context too big. Unable to parse jobs.")
    return res

def main_preprocess(name):
    preprocess_post(f"data\{name}_posts.json",f"enriched_data\{name}_posts.json")

if __name__=="__main__":
    name=["warikoo","kunalshah1","garyvaynerchuk","justinwelsh","swati-bathwal-211052143","sahilbloom","morgan-housel-5b473821"]
    for a in name:
        preprocess_post(f"data\{a}_posts.json",f"enriched_data\{a}_posts.json")