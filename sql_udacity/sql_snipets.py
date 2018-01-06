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
