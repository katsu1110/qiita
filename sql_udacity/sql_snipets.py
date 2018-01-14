# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:44:57 2018

@author: katsuhisa
"""

'''code snipets used in the online lecture 'Intro to Relational Databases' by Udacity'''
### Lesson 1 'Data and Tables'
# Contents ---------------------
# Tables, Aggregations, Queries, Keys, Join

Query = '''
select food from diet where species = 'oranghutan';
select 2+2 as sum;
'''

### Lesson 2 'Elements of SQL'
# Contents ---------------------
# SQL types, Operators, Querry, select, insert, join, having

Query = '''
select name, birthdate from animals where species = 'gorilla';
select name, birthdate from animals where species = 'gorilla' and name = 'Max';
select name, birthdate from animals where species != 'gorilla' and name != 'Max';
select name from animals where species = 'llamas' and birthdate >= '1995-01-01' and birthdate <= '1998-12-31';
select max(name) from animals;
select * from animals limit 10;
select * from animals where species = 'orngutan' order by birthdate;
select name from animals where species = 'orngutan' order by birthdate desc;
select name, birthdate from animals order by name limit 10 offset 20;
select species, min(birthdate) from animals group by species; 
select count(*) as num, species from animals group by species order by num desc;
insert into animals values('Wibble','opossum','2018-01-06');
select name from animals, diet where animals.species = diets.species and diet.food = 'fish';
select name, count(*) as num from sales having num > 5;
select food, count(animals.name) as num from diet join animals on diet.species = animals.species group by food having num=1;
select food, count(animals.name) as num from diet, animals where diet.species = animals.species group by food having num=1;
select ordernames.name, count(*) as num from animals, taxonomy, ordernames 
where animals.species = taxonomy.name and taxonomy.t_order = ordernames.t_order
group by ordernames.name order by num desc;
'''
### Lesson 3 'Python DB-API'
'''Use "winpty vagrant ssh" !'''
# Contents ----------------------
# Writing code with DB API ------------------------
import sqlite3
conn = sqlite3.connect("Cookies")
cursor = conn.cursor()
cursor.execute(
        "select host_key from cookies limit 10")
results = cursor.fetchall()
print(results)
conn.close()

# Trying out DB API ---------------------------------------
import sqlite3

# Fetch some student records from the database.
db = sqlite3.connect("students")
c = db.cursor()
query = "select name, id from students order by name;"
c.execute(query)
rows = c.fetchall()

# First, what data structure did we get?
print("Row data:")
print(rows)

# And let's loop over it too:
print("Student names:")
for row in rows:
  print ("  ", row[0])

db.close()

# inserts in DB API
db.commit()   # for atomicity!
db.close()

# Hello PostgreSQL ------------------------------------------
# \dt ... list all the tables in the database
# \dt+ ... list tables plus additional information 
# \H ... switch between printing tables in plain text vs HTML
# select * from posts \watch  ... display the contents of the table
# and refresh it every two seconds

# Update, Delete
import psycopg2

DBNAME = "forum"

def get_posts():
  """Return all posts from the 'database', most recent first."""
  db = psycopg2.connect(database=DBNAME)
  c = db.cursor()
  c.execute("select content, time from posts order by time desc;")
  c.execute("update posts set content = 'cheese' where content like '%spam%';")
  c.execute("delete from posts where content = 'cheese';")
  return c.fetchall()
  db.close()
  
def add_post(content):
  """Add a post to the 'database' with the current timestamp."""
  db = psycopg2.connect(database=DBNAME)
  c = db.cursor()
  c.execute("insert into posts values ('$s')", (content,))
  db.commit()
  db.close()

### Lesson 4 'Deeper into SQL'
# normlized design
# 1) every row has the same number of columns
# 2) there is a unique key, and everything in a row says something about the key
# 3) facts that don't relate to the key belong in different tables
# 4) tables shouldn't imply relationships between columns
    
# create table and types
# create database name ['options];
# drop database name ['options];
