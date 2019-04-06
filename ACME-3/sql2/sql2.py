import sqlite3 as sql
import pandas as pd
import csv

def createTables():
    db = sql.connect("database")
    cur = db.cursor()

    cur.execute("DROP TABLE IF EXISTS classes;")
    cur.execute("DROP TABLE IF EXISTS grades;")
    cur.execute("DROP TABLE IF EXISTS fields;")
    cur.execute("DROP TABLE IF EXISTS students;")

    cur.execute("CREATE TABLE classes (id INT NOT NULL, name TEXT);")    
    cur.execute("CREATE TABLE grades (id INT NOT NULL, class_id INT, grade TEXT);")
    cur.execute("CREATE TABLE fields (id INT NOT NULL, name TEXT);")
    cur.execute("CREATE TABLE students (id INT NOT NULL, name TEXT, major INT, minor INT);")

    with open('classes.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
        cur.executemany("INSERT INTO classes VALUES (?, ?);", rows)

    with open('grades.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
        cur.executemany("INSERT INTO grades VALUES (?, ?, ?);", rows)

    with open('fields.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
        cur.executemany("INSERT INTO fields VALUES (?, ?);", rows)

    with open('students.csv', 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=',')]
        cur.executemany("INSERT INTO students VALUES (?, ?, ?, ?);", rows)

    db.commit()
    db.close()



def prob1():
    """
    Specify relationships between columns in given sql tables.
    """
    print "One-to-one relationships:"
    # Put print statements specifying one-to-one relationships between table
    # columns.
    print "Student ID ---> Name"
    print "Course ID ---> Course Name"
    print "Major ID ---> Major Name"

    print "**************************"
    print "One-to-many relationships:"
    # Put print statements specifying one-to-many relationships between table
    # columns.
    print "Major ID/Major Name ---> Student ID/Student Name"
    print "Minor ID/Minor Name ---> Student ID/Student Name"

    print "***************************"
    print "Many-to-Many relationships:"
    # Put print statements specifying many-to-many relationships between table
    # columns.
    print "Course ID ---> Student ID"
    print "Course Grade ---> Student ID"

def prob2():
    """
    Write a SQL query that will output how many students belong to each major,
    including students who don't have a major.

    Return: A table indicating how many students belong to each major.
    """
    createTables()
    #Build your tables and/or query here
    query = "SELECT fields.name, COUNT(students.id) FROM students LEFT OUTER JOIN fields ON students.major=fields.id GROUP BY fields.name ORDER BY fields.name;"
    db = sql.connect("database")
    cur = db.cursor() 
    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    result =  pd.read_sql_query(query, db)

    db.commit()
    db.close()
    return result


def prob3():
    """
    Select students who received two or more non-Null grades in their classes.

    Return: A table of the students' names and the grades each received.
    """
    createTables()
    #Build your tables and/or query here
    db = sql.connect("database")
    cur = db.cursor()
    
    query = "SELECT students.name, COUNT(grades.grade) FROM students LEFT OUTER JOIN grades ON students.id=grades.id GROUP BY students.name HAVING COUNT(grades.grade) >= 2;"
    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    result =  pd.read_sql_query(query, db)

    db.commit()
    db.close()
    return result


def prob4():
    """
    Get the average GPA at the school using the given tables.

    Return: A float representing the average GPA, rounded to 2 decimal places.
    """
    createTables()
    #Build your tables and/or query here
    db = sql.connect("database")
    cur = db.cursor()
    
    query = "SELECT students.name, COUNT(grades.grade) FROM students LEFT OUTER JOIN grades ON students.id=grades.id GROUP BY students.name HAVING COUNT(grades.grade) >= 2;"

    cur.execute("SELECT ROUND(AVG(points),2) FROM (SELECT CASE WHEN grade IN ('A+','A','A-') THEN 4.0 WHEN grade IN ('B+','B','B-') THEN 3.0 WHEN grade IN ('C+','C','C-') THEN 2.0 WHEN grade IN ('D+','D','D-') THEN 1.0 END AS points FROM grades WHERE grade IS NOT NULL);")

    result =  cur.fetchone()[0]

    db.commit()
    db.close()
    return result


def prob5():
    """
    Find all students whose last name begins with 'C' and their majors.

    Return: A table containing the names of the students and their majors.
    """
    createTables()
    #Build your tables and/or query here
    db = sql.connect("database")
    cur = db.cursor()

    query  = "SELECT students.name, fields.name FROM students LEFT OUTER JOIN fields ON students.major=fields.id WHERE students.name LIKE '% C%';"

    # This line will make a pretty table with the results of your query.
        ### query is a string containing your sql query
        ### db is a sql database connection
    result =  pd.read_sql_query(query, db)

    db.commit()
    db.close()
    return result

if __name__ == '__main__':
    pass
