# 노트북 안에서 그래프를 그리기 위해
%matplotlib inline
pwd

#Render Matplotlib Plots Inline
%matplotlib inline

#Import the standard Python Scientific Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#Import Plotly and use it in the Offline Mode
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

#Suppress Deprecation and Incorrect Usage Warnings
import warnings
warnings.filterwarnings('ignore')





# Import the standard Python Scientific Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress Deprecation and Incorrect Usage Warnings
import warnings
warnings.filterwarnings('ignore')

question = pd.read_csv('data/kaggle-survey-2017/schema.csv')
question.shape

question.tail()


# 판다스로 선다형 객관식 문제에 대한 응답을 가져옴
mcq = pd.read_csv('/Users/dongweonshin/Documents/DataScience/kaggle-survey-2017/multipleChoiceResponses.csv',
 encoding="ISO-8859-1", low_memory=False)
mcq.shape



#Load Free Form Responses into a Pandas DataFrame
ff = pd.read_csv('/Users/dongweonshin/Documents/DataScience/kaggle-survey-2017/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)
ff.shape


#The Seaborn Countplot function counts the number of instances of each category and renders a barplot.
sns.countplot(y='GenderSelect', data=mcq)


#Create a DataFrame for number of respondents by country
con_df = pd.DataFrame(mcq['Country'].value_counts())


mcq['Country']

mcq['Country'].value_counts()

con_df.index

con_df['country'] = con_df.index


con_df.columns = ['num_resp', 'country']

con_df

con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(10)


#Create a Choropleth Map of the respondents using Plotly.
#Find out more at https://plot.ly/python/choropleth-maps/
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_resp'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents'),
      ) ]

layout = dict(
    title = 'Survey Respondents by Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')



#Get Summary Statistics of the Respndents' Ages.
mcq['Age'].describe()

fig = fig_fact.create_distplot([mcq[mcq['Age'] > 0]['Age']], ['age'], colors=['#BA68C8'])
py.iplot(fig, filename='Basic Distplot')
#sns.distplot(mcq[mcq['Age'] > 0]['Age'])





sns.countplot(y='FormalEducation', data=mcq)


plt.figure(figsize=(6,8))
sns.countplot(y='MajorSelect', data=mcq)


sns.countplot(y='EmploymentStatus', data=mcq)





sns.countplot(y='Tenure', data=mcq)

sns.countplot(y='LanguageRecommendationSelect', data=mcq)


top_lang = mcq['LanguageRecommendationSelect'].value_counts()
top_lang_dist = []
for lang in top_lang.index:
    top_lang_dist.append(mcq[(mcq['Age'].notnull()) & (mcq['LanguageRecommendationSelect'] == lang)]['Age'])

group_labels = top_lang.index

fig = fig_fact.create_distplot(top_lang_dist, group_labels, show_hist=False)
py.iplot(fig, filename='Language Preferences by Age')



mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape

#Plot the number of R and Python users by Occupation
data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & ((mcq['LanguageRecommendationSelect'] == 'Python') | (mcq['LanguageRecommendationSelect'] == 'SAS') | (mcq['LanguageRecommendationSelect'] == 'R')
)]
plt.figure(figsize=(8, 10))
sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)





#Render a bar plot of the 15 most popular ML Tools for next year
data = mcq['MLToolNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)


data = mcq['MLMethodNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)

#Explode the Pandas Dataframe to get the number of times each Learning Platform was mentioned
mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))
s = mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'platform'



plt.figure(figsize=(6,8))
data = s[s != 'nan'].value_counts()
sns.barplot(y=data.index, x=data)
