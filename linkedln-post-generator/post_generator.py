from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from few_shot import FewShotPost

def generate_post(length,language,tags,selected_influencer):
    prompt_template, variables = get_prompt(length, language, tags, selected_influencer)
    formatted_prompt = prompt_template.format(**variables)
    response = llm.invoke(formatted_prompt)
    return response.content


def get_prompt(length,language,tags,selected_influencer):
    length_str=get_length_str(length)

    template = f'''
    Generate a LinkedIn post using the below information. No preamble.

    1) Topic: {tags}
    2) Length: {length_str}
    3) Language: {language}
    If Language is Hinglish then it means it is a mix of Hindi and English. 
    The script for the generated post should always be English.
    '''
    prompt=PromptTemplate.from_template(template)

    fs=FewShotPost(selected_influencer)
    example=fs.get_post(length,language,tags)
    
    if len(example)>0:
        prompt += "4) Use the writing style as per the following examples."

    for i,post in enumerate(example):
        prompt+=f'\n\nExample{i+1}: \n\n{post}'

        if i==2:
            break

    return prompt, {
        "tags": tags,
        "length_str": length_str,
        "language": language,
        "examples": example
    }

def get_length_str(length):
    if length=='Short':
        return '1 to 3 lines'
    if length=='Medium':
        return '5 to 7 lines'
    if length=='Long':
        return '9 to 13 lines'
    
if __name__ == "__main__":
    print(generate_post("Medium", "English", "self Improvement","warikoo"))