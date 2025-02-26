public class Player {
    public static final Vec2 DIMENSION = new Vec2(1000, 1000);
    //(xUnits/tick^2, yUnits/tick^2)
    public static final Vec2 MOVE_ACCEL = new Vec2(3, 0);
    public static final Vec2 JUMP_ACCEL = new Vec2(0, 350);
    public static final Vec2 GRAV_ACCEL = new Vec2(0, -15);
    public static final int SNAP_UP_THRESHOLD = 1000;
    //(xUnits/tick, yUnits/tick)
    Vec2 velocity;
    //(xUnits, yUnits)
    Vec2 position;
    // 0-freefall 1-grounded 2-dead
    int state;
    public Player() {
        position = new Vec2(0, 0);
        velocity = new Vec2(0, 0);
        state = 0;
    }
    public String toString() {
        String s = "State: "+state;
        s += "\nPosition: "+position.toString();
        return s + "\nVelocity: "+velocity.toString();
    }
    public boolean camCollisionCheck(int x, int y) {
        int playLeft = position.x;
        int playRight = position.x + DIMENSION.x;
        int playDown = position.y;
        int playUp = position.y + DIMENSION.y;
        return x >= playLeft && x <= playRight && y >= playDown && y <= playUp;
    }
    // index 0> 1< 2^
    public void handleVelocity(boolean[] input) {
        if(input[0] && state != 2) velocity = velocity.add(MOVE_ACCEL);
        if(input[1] && state != 2) velocity = velocity.add(MOVE_ACCEL.sMultiply(-1));
        if(input[2] && state == 1) {
            velocity = velocity.add(JUMP_ACCEL);
            state = 0;
        }
        if(state == 0) velocity = velocity.add(GRAV_ACCEL);
        velocity.x = Math.min(velocity.x, 200);
        velocity.x = Math.max(velocity.x, -200);
        velocity.y = Math.min(velocity.y, 500);
        velocity.y = Math.max(velocity.y, -500);
    }
    
    public void handlePosition() {
        position = position.add(velocity);
    }
}
