public abstract class GameObject {
    public final Vec2 dimension;
    public boolean isVisible, isTangible;
    public boolean collisionCheck(Vec2 position) {
        
    }
    public abstract void handleCollision();
}
