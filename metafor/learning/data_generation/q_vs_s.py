from collections import deque  # (this line is needed if not already imported)

class FCFSQueue:
    def __init__(self):
        self.deque = deque()
    
    def append(self, job):
        self.deque.append(job)

    def pop(self):
        return self.deque.popleft()

    def len(self) -> int:
        return len(self.deque)

    @staticmethod
    def name() -> str:
        return "FCFS"

q = FCFSQueue()
q.append("Job 1")
q.append("Job 2")
q.append("Job 3")
q.append("Job 4")
q.append("Job 5")
print(q.pop())  # Output: Job 1
print(q.len())  # Output: 1
print(FCFSQueue.name())  # Output: FCFS


from collections import deque

class LIFOStack:
    def __init__(self):
        self.deque = deque()

    def push(self, job):
        """Add a job to the top of the stack."""
        self.deque.append(job)

    def pop(self):
        """Remove and return the job from the top of the stack."""
        return self.deque.pop()

    def len(self) -> int:
        """Return the number of jobs in the stack."""
        return len(self.deque)

    @staticmethod
    def name() -> str:
        """Return the name of this scheduling discipline."""
        return "LIFO"

s = LIFOStack()
s.push("Job A")
s.push("Job B")
s.push("Job C")
s.push("Job D")
s.push("Job E")
print(s.pop())  # Output: Job B
print(s.len())  # Output: 1
print(LIFOStack.name())  # Output: LIFO