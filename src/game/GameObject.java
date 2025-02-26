// class representing an object that's not the player
public class GameObject {
    // vectors for size and location
    public final Vec2 dimension, position;
    // booleans for visibility, tangibility, and whether the object kills the player
    public boolean isVisible, isTangible, isDeathPlane;
    public GameObject(Vec2 p, Vec2 d) {
        dimension = d;
        position = p;
        // default state
        isVisible = true;
        isTangible = true;
        isDeathPlane = false;
    }
    // determines whether a point is contained within the object(for rasterization)
    public boolean camCollisionCheck(int x, int y) {
        int objLeft = position.x;
        int objRight = position.x + dimension.x;
        int objDown = position.y;
        int objUp = position.y + dimension.y;
        return x >= objLeft && x <= objRight && y >= objDown && y <= objUp;
    }
    // basic AABB check to determine player collision
    public boolean collisionCheck(Vec2 playerPos) {
        if(!isTangible) return false;
        int playLeft = playerPos.x;
        int playRight = playerPos.x+Player.DIMENSION.x;
        int playDown = playerPos.y;
        int playUp = playerPos.y + Player.DIMENSION.y;
        int objLeft = position.x;
        int objRight = position.x + dimension.x;
        int objDown = position.y;
        int objUp = position.y + dimension.y;
        boolean withinXBounds = (playLeft >= objLeft && playLeft <= objRight)
        || (playRight <= objRight && playRight >= objLeft);
        boolean withinYBounds = (playDown >= objDown && playDown <= objUp) 
        || (playUp <= objUp && playUp >= objDown);
        return withinXBounds && withinYBounds;
    }
    // if the player is inside of a block, handles how the player should be moved outside of it
    public Vec2 getDisplacement(Vec2 playerPos) {
        int playLeft = playerPos.x;
        int playRight = playerPos.x+Player.DIMENSION.x;
        int playDown = playerPos.y;
        int playUp = playerPos.y + Player.DIMENSION.y;
        int objLeft = position.x;
        int objRight = position.x + dimension.x;
        int objDown = position.y;
        int objUp = position.y + dimension.y;

        // chooses whatever movement is the smallest to push the player
        int yUp = objUp-playDown;
        // TODO treat yUP like all others
        if(yUp < Player.SNAP_UP_THRESHOLD) return new Vec2(0, yUp);
        int xLeft = playRight-objLeft;
        int yDown = playUp-objDown;
        int xRight = objRight-playLeft;
        if(xLeft < xRight && xLeft < yDown) return new Vec2(-1*xLeft, 0);
        else if(xRight < xLeft && xRight < yDown) return new Vec2(xRight, 0);
        else return new Vec2(0, -1*yDown);
    }
}
