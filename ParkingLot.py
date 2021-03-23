'''
Design Parking Lot System
https://leetcode.com/problems/design-parking-system/

'''

'''

1. Domain 

    Entities:
    1. Car : CarSize
        - CarSize size
    
    # or this can be enum
    2 CarSize
        - small = 1
        - medium = 2
        - large = 3
    
    2. ParkingLot
        - int capacity_small
        - int capacity_medium
        - int capacity_large
    

2. Application

    AddCar()
    
3. Data / repositories

4. Intrastructure

'''
import enum
class CarSize(enum.Enum):
    small = 1
    medium = 2
    large = 3

class ParkingLot:
    def __init__(self, small, medium, large):
        self.parkinglot = {
            CarSize.small.value: small,
            CarSize.medium: medium,
            CarSize.large: large
        }

    def addCar(self, size):
        if self.parkinglot[size]:
            self.parkinglot[size]-=1
            return True
        return False
