from abc import ABC, abstractmethod


class Person(ABC):
    def __init__(self, name, yob):
        self.__name = name
        self.__yob = yob

    @abstractmethod
    def describe(self):
        pass

    def get_yob(self):
        return self.__yob


class Ward:
    def __init__(self, name):
        self.__name = name
        self.__people = []

    def add_person(self, person):
        self.__people.append(person)

    def describe(self):
        print(f"Ward: {self.__name}")
        for person in self.__people:
            person.describe()

    def count_doctor(self):
        count = 0
        for person in self.__people:
            if isinstance(person, Doctor):
                count += 1
        return count

    def sort_age(self):
        self.__people.sort(key=lambda x: x.get_yob(), reverse=True)

    def compute_average(self):
        total_yob = 0
        count = 0
        for person in self.__people:
            if isinstance(person, Teacher):
                total_yob += person.yob
                count += 1
        return total_yob/count


class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self.__grade = grade

    def describe(self):
        print(
            f"Student - Name: {self.__name} - YOB: {self.__yob} - Grade: {self.__grade}")


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self.__subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self.__name} - YOB: {self.__yob} - Subject: {self.__subject}")


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self.__specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self.__name} - YOB: {self.__yob} - Specialist: {self.__specialist}")


ward = Ward("Ward1")
ward.add_person(Student(name="studentA", yob=2010, grade="7"))
ward.add_person(Teacher(name="teacherA", yob=1969, subject="Math"))
ward.add_person(Teacher(name="teacherB", yob=1995, subject="History"))
ward.add_person(Doctor(name="doctorB", yob=1975, specialist="Cardiologists"))
ward.add_person(Doctor(name="doctorA", yob=1945,
                specialist="Endocrinologists"))

ward.sort_age()
ward.describe()

print(f"Số lượng bác sĩ trong Ward: {ward.count_doctor()}")
print(f"Tuổi trung bình của giáo viên trong Ward: {ward.compute_average()}")
