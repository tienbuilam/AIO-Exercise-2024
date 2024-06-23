class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def is_empty(self):
        if len(self.__queue) == 0:
            return True
        return False

    def is_full(self):
        if len(self.__queue) == self.__capacity:
            return True
        return False

    def dequeue(self):
        if self.is_empty():
            return "Queue is empty"
        return self.__queue.pop(0)

    def enqueue(self, item):
        if self.is_full():
            return "Queue is full"
        self.__queue.append(item)

    def front(self):
        if self.is_empty():
            return "Queue is empty"
        return self.__queue[0]


queue1 = MyQueue(capacity=5)

queue1.enqueue(1)
queue1.enqueue(2)

print(queue1.is_full())
print(queue1.front())
print(queue1.dequeue())
print(queue1.front())
print(queue1.dequeue())

print(queue1.is_empty())
