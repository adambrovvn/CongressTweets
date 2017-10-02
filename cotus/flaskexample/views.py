from flask import render_template
from flaskexample import app
import pandas as pd


@app.route('/')
@app.route('/index')
def index():
	week=pd.read_csv('/Users/adambrown/AnacondaProjects/cotus/flaskexample/static/dfs/week.csv',sep=',',engine='python')
	indate=week['model'].iloc[0]
	total=len(week)
	uni=len(week['name'].drop_duplicates())
	y=week['lda_topic'].value_counts().tolist()
	x=week['lda_topic'].value_counts().index.tolist()
	plot_top=['Topic'+str(i) for i in x]
	odf = pd.DataFrame({'x':x,'y':y,'topic':plot_top})
	df=odf.sort_values(by='y',ascending=False)[:5]
	slist=df['x'].tolist()
	rc1=week[week['lda_topic']==slist[0]]
	rc1=rc1[['name','state','party','chamber']].drop_duplicates().sort_values(by=['state','chamber'])
	roll_one=[]
	for i in range(0,len(rc1)):
		roll_one.append(dict(name=rc1.name.iloc[i], state=rc1.state.iloc[i], party=rc1.party.iloc[i], chamber=rc1.chamber.iloc[i]))
	rc2=week[week['lda_topic']==slist[1]]
	rc2=rc2[['name','state','party','chamber']].drop_duplicates().sort_values(by=['state','chamber'])
	roll_two=[]
	for i in range(0, len(rc2)):
		roll_two.append(dict(name=rc2.name.iloc[i], state=rc2.state.iloc[i], party=rc2.party.iloc[i],chamber=rc2.chamber.iloc[i]))
	rc3=week[week['lda_topic']==slist[2]]
	rc3=rc3[['name','state','party','chamber']].drop_duplicates().sort_values(by=['state','chamber'])
	roll_three=[]
	for i in range(0, len(rc3)):
		roll_three.append(dict(name=rc3.name.iloc[i], state=rc3.state.iloc[i], party=rc3.party.iloc[i],chamber=rc3.chamber.iloc[i]))
	rc4=week[week['lda_topic']==slist[3]]
	rc4=rc4[['name','state','party','chamber']].drop_duplicates().sort_values(by=['state','chamber'])
	roll_four=[]
	for i in range(0, len(rc4)):
		roll_four.append(dict(name=rc4.name.iloc[i], state=rc4.state.iloc[i], party=rc4.party.iloc[i], chamber=rc4.chamber.iloc[i]))
	rc5=week[week['lda_topic']==slist[4]]
	rc5=rc5[['name','state','party','chamber']].drop_duplicates().sort_values(by=['state','chamber'])
	roll_five=[]
	for i in range(0, len(rc5)):
		roll_five.append(dict(name=rc5.name.iloc[i], state=rc5.state.iloc[i], party=rc5.party.iloc[i], chamber=rc5.chamber.iloc[i]))
	return render_template("single-post.html",total=total,roll_one=roll_one,uni=uni,roll_two=roll_two,roll_three=roll_three,roll_four=roll_four,roll_five=roll_five,indate=indate)

@app.route('/about')
def how():
	return render_template('about.html')

@app.route('/db')
def birth_page():
    sql_query = """                                                             
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
;                                                                               
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    print(query_results[:10])
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)
