public class GameObject {
    public final Vec2 dimension, position;
    public boolean isVisible, isTangible, isDeathPlane;
    public GameObject(Vec2 p, Vec2 d) {
        dimension = d;
        position = p;
        isVisible = true;
        isTangible = true;
        isDeathPlane = false;
    }
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
    public Vec2 getDisplacement(Vec2 playerPos) {
        int playLeft = playerPos.x;
        int playRight = playerPos.x+Player.DIMENSION.x;
        int playDown = playerPos.y;
        int playUp = playerPos.y + Player.DIMENSION.y;
        int objLeft = position.x;
        int objRight = position.x + dimension.x;
        int objDown = position.y;
        int objUp = position.y + dimension.y;
        
        int xLeft = playRight-objLeft;
        int yDown = playUp-objDown;
        int xRight = objRight-playLeft;
        if(xLeft < xRight && xLeft < yDown) return new Vec2(-1*xLeft, 0);
        else if(xRight < xLeft && xRight < yDown) return new Vec2(xRight, 0);
        else return new Vec2(0, -1*yDown);
    }
}
