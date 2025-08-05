import json
import pandas as pd

class FewShotPost():
    def __init__(self,selected_influencer):
        self.unique_tags=None
        self.df=None
        file_path=f'enriched_data\{selected_influencer}_posts.json'
        self.load_post(file_path)

    def load_post(self,file_path):
        with open (file_path,encoding='utf-8') as file:
            posts=json.load(file)
            self.df=pd.json_normalize(posts)
            self.df['Length']=self.df['line_count'].apply(self.get_length)
            all_tags = self.df['tags'].apply(lambda x: x).sum()
            self.unique_tags = list(set(all_tags))

    def get_post(self,language,length,tag):
        df_filtered=self.df[
        (self.df['language']==language)&
        (self.df['Length']==length)&
        (self.df['tags'].apply(lambda tags: tag in tags))
        ]
        return df_filtered.to_dict(orient='records')

    def get_length(self,line_count):
        if line_count<5:
            return "Short"
        elif 5 <= line_count < 9:
            return "Medium" 
        else:
            return "Long"
    def get_tags(self):
        return self.unique_tags   

if __name__=="__main__":
    fs=FewShotPost()
    post=fs.get_post("English","Long","Motivation")
    print(post)