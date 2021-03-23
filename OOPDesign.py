'''
Steps to take:

1. Clarify requirement. think about entities
2. Relation between them.
3. Write them down. Assume payement services, database services are implement.

Before writing think about:

1) Encapsulation: Means binding the data together in objects.

2) Abstraction: Means hiding all but the relevant data about an object, this will help reduce the system complexity.

3) Inheritance: Means making new classes from already defined classes.

4) Polymorphism: The ability of different methods to implement the same operation differently.


Begin:

Library Management System
https://www.educative.io/courses/grokking-the-object-oriented-design-interview/RMlM3NgjAyR

1. Define constants :
    1. Enum classes for states
        class ReservationStatus(Enum):
            WAITING, PENDING, CANCELED, NONE = 1, 2, 3, 4

    2. Properties
        class Person(ABC):
          def __init__(self, name, address, email, phone):
            self.__name = name
            self.__address = address
            self.__email = email
            self.__phone = phone

    3. Constant
        class Constants:
          self.MAX_BOOKS_ISSUED_TO_A_USER = 5
          self.MAX_LENDING_DAYS = 10

2. To Inherit

    import abc
    class Account(ABC):
      def __init__(self, id, password, person, status=AccountStatus.Active):
        self.__id = id
        self.__password = password
        self.__status = status
        self.__person = person

      def reset_password(self):
        None


    class Librarian(Account):
      def __init__(self, id, password, person, status=AccountStatus.Active):
        super().__init__(id, password, person, status)

      def add_book_item(self, book_item):
        None

      def block_member(self, member):
        None

      def un_block_member(self, member):
        None





'''

class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        """
        self.food = food
        self.height = height
        self.width = width
        self.snake = collections.deque([(0, 0)])
        self.score = 0

    def move(self, direction: str) -> int:
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
        @return The game's score after the move. Return -1 if game over.
        Game over when snake crosses the screen boundary or bites its body.
        """
        current_row, current_column = self.snake[-1]
        if direction == "U":
            current_row -= 1
        elif direction == "D":
            current_row += 1
        elif direction == "L":
            current_column -= 1
        elif direction == "R":
            current_column += 1

        if 0 > current_row or current_row > self.height - 1 or 0 > current_column or current_column > self.width - 1:
            return -1

        if self.food and [current_row, current_column] == self.food[0]:
            self.snake.append([current_row, current_column])
            self.food.pop(0)
            self.score += 1
        else:
            self.snake.popleft()

            if [current_row, current_column] in self.snake:
                return -1
            else:
                self.snake.append([current_row, current_column])

        return self.score


# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)


'''
Interfaces in Python

'''

import abc


class MyInterface(abc.ABC):

    @abs.absstactmethod
    def method(self):
        pass


class MyClass(MyInterface):

    def method(self):
        print("overridden!")

'''
Design File System

'''


# The TrieNode data structure.
class TrieNode(object):
    def __init__(self, name):
        self.map = defaultdict(TrieNode)
        self.name = name
        self.value = -1


class FileSystem:

    def __init__(self):

        # Root node contains the empty string.
        self.root = TrieNode("")

    def createPath(self, path: str, value: int) -> bool:

        # Obtain all the components
        components = path.split("/")

        # Start "curr" from the root node.
        cur = self.root

        # Iterate over all the components.
        for i in range(1, len(components)):
            name = components[i]

            # For each component, we check if it exists in the current node's dictionary.
            if name not in cur.map:

                # If it doesn't and it is the last node, add it to the Trie.
                if i == len(components) - 1:
                    cur.map[name] = TrieNode(name)
                else:
                    return False
            cur = cur.map[name]

        # Value not equal to -1 means the path already exists in the trie.
        if cur.value != -1:
            return False

        cur.value = value
        return True

    def get(self, path: str) -> int:

        # Obtain all the components
        cur = self.root

        # Start "curr" from the root node.
        components = path.split("/")

        # Iterate over all the components.
        for i in range(1, len(components)):

            # For each component, we check if it exists in the current node's dictionary.
            name = components[i]
            if name not in cur.map:
                return -1
            cur = cur.map[name]
        return cur.value

'''
evaluate expression tree
https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/

'''


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def evaluate(self) -> int:
        if self.val in ["+", "-", "*", "/"]:
            l = self.left.evaluate()
            r = self.right.evaluate()
            if self.val == '+':
                return l + r
            elif self.val == '-':
                return l - r
            elif self.val == '*':
                return l * r
            else:
                return l // r
        else:
            return self.val


class TreeBuilder(object):
    def buildTree(self, postfix: List[str]) -> 'Node':
        item = postfix.pop()
        root = TreeNode(int(item)) if item.isdigit() else TreeNode(item)
        if item in ["+", "-", "*", "/"]:
            root.right = self.buildTree(postfix)
            root.left = self.buildTree(postfix)
            return root
        else:
            return root
