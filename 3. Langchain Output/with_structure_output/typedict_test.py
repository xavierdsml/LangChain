from typing import TypedDict

class Person(TypedDict):
  name : str
  age : int

new_person : Person = {'name' : 'tushar', 'age' : '21'}
print(new_person)