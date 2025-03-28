# Game Internals
import random

class Player():
    WIDTH = 1000
    HEIGHT = 1000
    MOVE_ACCEL = 6
    JUMP_ACCEL = 350
    GRAV_ACCEL = 15
    FRIC_ACCEL = 3

    def __init__(self):
        self.position_x = 0
        self.position_y = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.state = 0

    def point_collision_check(self, x, y):
        player_left = self.position_x
        player_right = self.position_x + Player.WIDTH
        player_bottom = self.position_y
        player_top = self.position_y + Player.HEIGHT
        return x >= player_left and x <= player_right and y >= player_bottom and y <= player_top
    
    def handle_movement(self, input):
        right_pressed = (input & 1) == 1
        left_pressed = (input>>1 & 1) == 1
        up_pressed = (input>>2 & 1) == 1

        if(right_pressed and self.state != 2):
            self.velocity_x += Player.MOVE_ACCEL
        if(left_pressed and self.state != 2):
            self.velocity_x -= Player.MOVE_ACCEL
        if(up_pressed and self.state == 1):
            self.velocity_y += Player.JUMP_ACCEL
            self.state = 0

        if(self.state == 0):
            self.velocity_y -= Player.GRAV_ACCEL
        if(self.velocity_x < 3 and self.velocity_x > -3):
            self.velocity_x = 0
        elif(self.velocity_x > 0):
            self.velocity_x -= Player.FRIC_ACCEL
        else:
            self.velocity_x += Player.FRIC_ACCEL

        self.velocity_x = min(self.velocity_x, 200)
        self.velocity_x = max(self.velocity_x, -200)
        self.velocity_y = min(self.velocity_y, 500)
        self.velocity_y = max(self.velocity_y, -500)

        self.position_x += self.velocity_x
        self.position_y += self.velocity_y

class GameObject():

    def __init__(self, x, y, width, height, is_visible=False, is_tangible=False, is_death_plane=False, is_winpad=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_visible = is_visible
        self.is_tangible = is_tangible
        self.is_death_plane = is_death_plane
        self.is_winpad = is_winpad

    def point_collision_check(self, point_x, point_y):
        object_left = self.x
        object_right = self.x + self.width
        object_bottom = self.y
        object_top = self.y + self.height
        return point_x >= object_left and point_x <= object_right and point_y >= object_bottom and point_y <= object_top
    
    def player_collision_check(self, player_x, player_y):
        if not self.is_tangible:
            return False
        player_left = player_x
        player_right = player_x+Player.WIDTH
        player_bottom = player_y
        player_top = player_y + Player.HEIGHT
        object_left = self.x
        object_right = self.x + self.width
        object_bottom = self.y
        object_top = self.y + self.height
        within_x_bounds = (player_left >= object_left and player_left <= object_right) or (player_right <= object_right and player_right >= object_left)
        within_y_bounds = (player_bottom >= object_bottom and player_bottom <= object_top) or (player_top <= object_top and player_top >= object_bottom)
        return within_x_bounds and within_y_bounds

    def get_player_displacement(self, player_x, player_y):
        player_left = player_x
        player_right = player_x+Player.WIDTH
        player_bottom = player_y
        player_top = player_y + Player.HEIGHT
        object_left = self.x
        object_right = self.x + self.width
        object_bottom = self.y
        object_top = self.y + self.height

        y_push_up = object_top-player_bottom+1
        x_push_left = player_right-object_left+1
        y_push_down = player_top-object_bottom+1
        x_push_right = object_right-player_left+1
        if(x_push_left < x_push_right and x_push_left < y_push_down and x_push_left < y_push_up):
             return (-1*x_push_left, 0)
        elif(x_push_right < y_push_down and x_push_right < y_push_up):
            return (x_push_right, 0)
        elif(y_push_down < y_push_up):
            return (0, -1*y_push_down)
        else:
            return (0, y_push_up)

class GameHandler():
    
    def __init__(self, player_start_x=0, player_start_y=0):
        self.player = Player()
        self.player.position_x = player_start_x
        self.player.position_y = player_start_y
        self.game_objects = []
        self.game_objects.append(GameObject(-500000, -10000, 1000000, 10000, True, True))

    def load_objects_from_text(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                object_values = line.split(",")
                is_visible = int(object_values[4]) & 1 == 1
                is_tangible = (int(object_values[4]) >> 1) & 1
                is_death_plane = (int(object_values[4]) >> 2) & 1
                is_winpad = (int(object_values[4]) >> 3) & 1
                object = GameObject(int(object_values[0]), int(object_values[1]), int(object_values[2]), int(object_values[3]), is_visible, is_tangible, is_death_plane, is_winpad)
                self.game_objects.append(object)

    def get_pixel_value(self, x, y):
        for object in self.game_objects:
            if not object.is_visible:
                continue
            if object.point_collision_check(x, y):
                if object.is_death_plane:
                    return 2
                elif object.is_winpad:
                    return 4
                return 1
        if self.player.point_collision_check(x, y):
            return 3
        return 0
    
    # 0 - death 1 - normal 2 - win
    def do_game_tick(self, input):
        if self.player.state == 2:
            return 0
        self.player.state = 0
        break_flag = False
        for i in range(0, 1001, 200):
            for object in self.game_objects:
                if not object.is_visible and not object.is_winpad:
                    continue
                if object.point_collision_check(self.player.position_x+i, self.player.position_y-1):
                    self.player.state = 1
                    break_flag = True
                    break
            if(break_flag):
                break
        
        self.player.handle_movement(input)
        for object in self.game_objects:
            if object.player_collision_check(self.player.position_x, self.player.position_y):
                if object.is_death_plane:
                    self.player.state = 2
                    return 0
                elif object.is_winpad:
                    self.player.state = 2
                    return 2
                player_displacement = object.get_player_displacement(self.player.position_x, self.player.position_y)
                if(player_displacement[1] == 0):
                    self.player.position_x += player_displacement[0]
                    self.player.velocity_x = 0
                elif(player_displacement[0] == 0):
                    self.player.position_y += player_displacement[1]
                    self.player.velocity_y = 0
                    if(player_displacement[1] > 0):
                        self.player.state = 1
        return True

# Graphics Handling

class DisplayHandler():

    def __init__(self, game, x_resolution=60, y_resolution=30, x_camera_range=16000, y_camera_range=14000):
        self.game = game
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.x_camera_range = x_camera_range
        self.y_camera_range = y_camera_range

    def get_pixel_buffer(self):
        camera_x_center = self.game.player.position_x + 500
        camera_y_center = self.game.player.position_y + 1000
        pixel_buffer = [[0 for i in range(0, self.y_resolution)] for i in range(0, self.x_resolution)]
        for i in range(0, self.x_resolution):
            for j in range(0, self.y_resolution):
                pixel_x = i*self.x_camera_range/self.x_resolution + camera_x_center - self.x_camera_range/2
                pixel_y = j*self.y_camera_range/self.y_resolution + camera_y_center - self.y_camera_range/2
                pixel_buffer[i][j] = self.game.get_pixel_value(pixel_x, pixel_y)
        return pixel_buffer

    def get_string_display(self):
        pixel_buffer = self.get_pixel_buffer()
        return_string = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        for j in range(self.y_resolution-1, -1, -1):
            for i in range(0, self.x_resolution):
                match pixel_buffer[i][j]:
                    case 0:
                        return_string += " "
                    case 1:
                        return_string += "o"
                    case 2:
                        return_string += "/"
                    case 3:
                        return_string += "z"
                    case 4:
                        return_string += "$"
            return_string += "\n"
        return return_string

# Environment
class Environment():

    def __init__(self, filepath):
        self.filepath = filepath
        self.reset()

    def reset(self, isXRandom=True):
        randomX = 15000
        while not((randomX < 14000) or (randomX > 20000 and randomX < 32000) or (randomX > 47000 and randomX < 53000) or (randomX > 65000 and randomX < 70000) or (randomX > 95000)):
            randomX = random.randint(2000, 105000)
        if not isXRandom:
            randomX = 4000

        self.game = GameHandler(randomX, 22000)
        self.game.load_objects_from_text(self.filepath)
        self.display = DisplayHandler(self.game)
        self.checkpoints = [False for i in range(0, 2000)]
        self.ticks_since_last_reward = 0

    def get_state(self):
        return self.display.get_pixel_buffer()

    def do_game_tick(self, input):
        # reward function
        checkpoint_start = (int)(self.game.player.position_x // 100)
        result = self.game.do_game_tick(input)
        if result == 0:
            return (-5000, True)
        elif result == 2:
            return (5000, True)
        checkpoint_end = (int)(self.game.player.position_x // 100)
        reward = 0
        while(checkpoint_start < checkpoint_end):
            if not self.checkpoints[checkpoint_start]:
                reward += 10
                self.checkpoints[checkpoint_start] = True
            checkpoint_start += 1
        if(reward == 0):
            self.ticks_since_last_reward += 1
        else:
            self.ticks_since_last_reward = 0
        if(self.ticks_since_last_reward <= 200):
            return (reward, False)
        else:
            return (-5000, True)
    
    def get_display(self):
        return self.display.get_string_display()