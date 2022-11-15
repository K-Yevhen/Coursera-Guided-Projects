from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc, col, max, struct
import matplotlib.pyplot as plts

spark = SparkSession.builder.appName('spark_app').getOrCreate()
listeting_csv_path = '/Users/yevhenkuts/PycharmProjects/pythonProject/CourseraGP/psprk/listenings.csv'
listeting_df = spark.read.format('csv').option('interSchema', True).option('header', True).load(listeting_csv_path)

# print(listeting_df.show())
listeting_df = listeting_df.drop('date')
listeting_df = listeting_df.na.drop()
# listeting_df.show()

# listeting_df.printSchema()
shape = (listeting_df.count(), listeting_df.columns)
# print(shape)

q0 = listeting_df.select('artist', 'track')
# q0.show()

q1 = listeting_df.select('*').filter(listeting_df.artist == 'Rihanna')
# q1.show()

q2 = listeting_df.select('user_id').filter(listeting_df.artist == 'Rihanna')
# q2.show()

q2 = listeting_df.select('user_id').filter(listeting_df.artist == 'Rihanna').groupby('user_id').agg(count('user_id').alias('count')).orderBy(desc('count')).limit(10)
# q2.show()

q3 = listeting_df.select('artist', 'track').groupby('artist', 'track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
# q3.show()

q4 = listeting_df.select('artist', 'track').filter(listeting_df.artist == 'Rihanna').groupby('artist', 'track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
# q4.show()

q5 = listeting_df.select('artist', 'album').groupby('artist', 'album').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
# q5.show()

genre_csv_path = '/Users/yevhenkuts/PycharmProjects/pythonProject/CourseraGP/psprk/genre.csv'
genre_df = spark.read.format('csv').option('inferSchema', True).option('header', True).load(genre_csv_path)
# genre_df.show()

data = listeting_df.join(genre_df, how='inner', on=['artist'])
# data.show()

q6 = data.select('user_id').filter(data.genre == 'pop').groupby('user_id').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
# q6.show()

q7 = data.select('genre').groupby('genre').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
# q7.show()

q8_1 = data.select('user_id', 'genre').groupby('user_id', 'genre').agg(count('*').alias('count')).orderBy('user_id')
# q8_1.show()

q8_2 = q8_1.groupby('user_id').agg(max(struct(col('count'),col('genre'))).alias('max'))
#q8_2.show()

q9 = genre_df.select("genre").filter( (col('genre') =='pop') | (col('genre') == 'metal') | (col('genre') == 'rock') | (col('genre') == 'hip hop')).groupby('genre').agg(count('genre').alias('count'))
#q9.show()

q9_list = q9.collect()

lables = [row['genre'] for row in q9_list]
counts = [row['count'] for row in q9_list]

print(lables)
print(counts)

plts.bar(lables, counts)
