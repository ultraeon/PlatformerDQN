# imports
import math

# Game Internals
class Player():
    WIDTH = 1000
    HEIGHT = 1000
    MOVE_ACCEL = 3
    JUMP_ACCEL = 350
    GRAV_ACCEL = 15
    FRIC_ACCEL = 0.98

    def __init__(self):
        self.posX = 0
        self.posY = 0
        self.velX = 0
        self.velY = 0
        self.state = 0

    def camCollisionCheck(self, x, y):
        playLeft = self.posX
        playRight = self.posX + Player.WIDTH
        playDown = self.posY
        playUp = self.posY + Player.HEIGHT
        return x >= playLeft and x <= playRight and y >= playDown and y <= playUp
    
    def handleMovement(self, input):
        rPressed = (input & 1) == 1
        lPressed = (input>>1 & 1) == 1
        uPressed = (input>>2 & 1) == 1

        if(rPressed and self.state != 2):
            self.velX += Player.MOVE_ACCEL
        if(lPressed and self.state != 2):
            self.velX -= Player.MOVE_ACCEL
        if(uPressed and self.state == 1):
            self.velY += Player.JUMP_ACCEL
            self.state = 0

        if(self.state == 0):
            self.velY -= Player.GRAV_ACCEL
        elif(self.state == 1):
            self.velX *= Player.FRIC_ACCEL

        self.velX = math.min(self.velX, 200)
        self.velX = math.max(self.velX, -200)
        self.velY = math.min(self.velY, 500)
        self.velY = math.max(self.velY, -500)

        self.posX += self.velX
        self.posY += self.velY

class GameObject():

    def __init__(self, x, y, width, height, isVisible=False, isTangible=False, isDeathPlane=False):
        self.posX = x
        self.posY = y
        self.width = width
        self.height = height
        self.isVisible = isVisible
        self.isTangible = isTangible
        self.isDeathPlane = isDeathPlane

    def camCollisionCheck(self, x, y):
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height
        return x >= objLeft and x <= objRight and y >= objDown and y <= objUp
    
    def collisionCheck(self, playerX, playerY):
        if not self.isTangible:
            return False
        playLeft = playerX
        playRight = playerX+Player.WIDTH
        playDown = playerY
        playUp = playerY + Player.HEIGHT
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height
        withinXBounds = (playLeft >= objLeft and playLeft <= objRight) or (playRight <= objRight and playRight >= objLeft)
        withinYBounds = (playDown >= objDown and playDown <= objUp) or (playUp <= objUp and playUp >= objDown)
        return withinXBounds and withinYBounds

    def getDisplacement(self, playerX, playerY):
        playLeft = playerX
        playRight = playerX+Player.WIDTH
        playDown = playerY
        playUp = playerY + Player.HEIGHT
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height

        yUp = objUp-playDown+1
        xLeft = playRight-objLeft+1
        yDown = playUp-objDown+1
        xRight = objRight-playLeft+1
        if(xLeft < xRight and xLeft < yDown and xLeft < yUp):
             return (-1*xLeft, 0)
        elif(xRight < yDown and xRight < yUp):
            return (xRight, 0)
        elif(yDown < yUp):
            return (0, -1*yDown)
        else:
            return (0, yUp)

class GameHandler():
    
    def __init__(self, x=0, y=0):
        self.player = Player()
        self.player.posX = x
        self.player.posY = y
        self.gameObjList = []
        self.gameObjList.append(GameObject(-500000, -10000, 1000000, 10000, True, True))

    def loadObjsFromText(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                vals = line.split(",")
                isV = int(vals[4]) & 1 == 1
                isT = (int(vals[4]) >> 1) & 1
                isDP = (int(vals[4]) >> 2) & 1
                obj = GameObject(int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), isV, isT, isDP)
                self.gameObjList.append(obj)

    def getCamCollision(self, x, y):
        for obj in self.gameObjList:
            if not obj.isVisible:
                continue
            if obj.camCollisionCheck(x, y):
                if obj.isDeathPlane:
                    return 2
                return 1
        if self.player.camCollisionCheck(x, y):
            return 3
        return 0
    
    def doTick(self, input):
        if self.player.state == 2:
            return False
        self.player.state = 0
        bFlag = False
        for i in range(0, 1001, 200):
            for obj in self.gameObjList:
                if not obj.isVisible:
                    continue
                if obj.camCollisionCheck(self.player.posX+i, self.player.posY-1):
                    self.player.state = 1
                    bFlag = True
                    break
            if(bFlag):
                break
        
        self.player.handleMovement(input)
        for obj in self.gameObjList:
            if obj.collisionCheck(self.player.posX, self.player.posY):
                if obj.isDeathPlane:
                    self.player.state = 2
                    return False
                displacement = obj.getDisplacement(self.player.posX, self.player.posY)
                if(displacement[2] == 0):
                    self.player.posX += displacement.x
                    self.player.velX = 0
                elif(displacement[1] == 0):
                    self.player.posY += displacement.y
                    self.player.velY = 0
                    if(displacement.y > 0):
                        self.player.state = 1
        return True

# Graphics Handling

class DisplayHandler():

    def __init__(self, game, xRes=60, yRes=30, xRange=14000, yRange=10000):
        self.game = game
        self.xResolution = xRes
        self.yResolution = yRes
        self.xRange = xRange
        self.yRange = yRange

    def getPixelBuffer(self):
        xPosition = self.game.player.posX+500
        yPosition = self.game.player.posY+3000
        pBuffer = [[0 for i in range(0, self.yResolution)] for i in range(0, self.xResolution)]
        for i in range(0, self.xResolution):
            for j in range(0, self.yResolution):
                camX = i*self.xRange/self.xResolution+xPosition-self.xRange/2
                camY = j*self.yRange/self.yResolution+yPosition-self.yRange/2
                pBuffer[i][j] = self.game.getCamCollision(camX, camY)
        return pBuffer

    def getStringDisplay(self):
        pBuffer = self.getPixelBuffer()
        s = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        for j in range(self.yResolution-1, -1, -1):
            for i in range(0, self.xResolution):
                match pBuffer[i][j]:
                    case 0:
                        s += " "
                    case 1:
                        s += "o"
                    case 2:
                        s += "/"
                    case 3:
                        s += "z"
            s += "\n"
        return s

class EnvironmentInfo():

    def __init__(self, filepath):
        self.filepath = filepath
        self.reset()

    def reset(self):
        self.game = GameHandler(1000, 11000)
        self.game.loadObjects(self.filepath)
        self.display = DisplayHandler(self.game)

    def getState(self):
        return self.display.getPixelBuffer()

    def doTick(self, input):
        p1 = self.game.player.posX
        if not self.game.doTick(input):
            return -50
        return self.game.player.posX-p1
    
game = GameHandler()
display = DisplayHandler(game)
print(display.getStringDisplay())