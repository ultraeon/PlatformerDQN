// class representing the player
public class Player {
    // important constants for global access
    public static final Vec2 DIMENSION = new Vec2(1000, 1000);
    // acceleration vectors as <xUnits/tick^2, yUnits/tick^2>
    public static final Vec2 MOVE_ACCEL = new Vec2(3, 0);
    public static final Vec2 JUMP_ACCEL = new Vec2(0, 350);
    public static final Vec2 GRAV_ACCEL = new Vec2(0, -15);
    public static final Vec2 FRIC_ACCEL = new Vec2(10, 0);
    // velocity vector as <xUnits/tick, yUnits/tick>
    Vec2 velocity;
    // position vector as <xUnits, yUnits>
    Vec2 position;
    // possible states 0->freefall 1->grounded 2->dead
    int state;
    public Player() {
        position = new Vec2(0, 0);
        velocity = new Vec2(0, 0);
        state = 0;
    }
    // prints out state, the position vector, and velocity vector
    public String toString() {
        String s = "State: "+state;
        s += "\nPosition: "+position.toString();
        return s + "\nVelocity: "+velocity.toString();
    }
    // checks whether a point is within the player(for rasterization)
    public boolean camCollisionCheck(int x, int y) {
        int playLeft = position.x;
        int playRight = position.x + DIMENSION.x;
        int playDown = position.y;
        int playUp = position.y + DIMENSION.y;
        return x >= playLeft && x <= playRight && y >= playDown && y <= playUp;
    }
    // input is whether arrow keys have been pressed in the form [>, <, ^] 
    public void handleVelocity(boolean[] input) {
        // changes velocity based on input
        if(input[0] && state != 2) velocity = velocity.add(MOVE_ACCEL);
        if(input[1] && state != 2) velocity = velocity.add(MOVE_ACCEL.sMultiply(-1));
        if(input[2] && state == 1) {
            velocity = velocity.add(JUMP_ACCEL);
            state = 0;
        }
        
        // adds gravity if player is in freefall
        if(state == 0) velocity = velocity.add(GRAV_ACCEL);
        else if(state == 1 && !input[0] && !input[1]) {
            if(velocity.x > FRIC_ACCEL.x) velocity.add(FRIC_ACCEL.sMultiply(-1));
            else if(velocity.x < -1*FRIC_ACCEL.x) velocity.add(FRIC_ACCEL);
            else velocity.x = 0;
        }
        // x-velocity bounded in [-200, 200] y-velocity bounded in [-500, 500]
        velocity.x = Math.min(velocity.x, 200);
        velocity.x = Math.max(velocity.x, -200);
        velocity.y = Math.min(velocity.y, 500);
        velocity.y = Math.max(velocity.y, -500);
    }
    // updates the position based on the current velocity
    public void handlePosition() {
        position = position.add(velocity);
    }
}
