from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
  name : str
  age : Optional[int] = None # default value
  mail : EmailStr
  cgpa : float = Field(gt = 0, lt = 10, default=18, description='A decimal value representing the CGPA of the student')

new_student = {'name':'tushar-gupta', 'mail':'tg@google.com', 'cgpa':5}
Student = Student(**new_student) #create object of the Student

print(Student)

student_dict = dict(Student)
print(student_dict['age'])

student_json = Student.model_dump_json()