import java.util.ArrayList;

public class GameHandler {
    private ArrayList<GameObject> gameObjList;
    private Player player;
    public GameHandler() {
        player = new Player();
        gameObjList = new ArrayList<GameObject>();
        gameObjList.add(new GameObject(new Vec2(-500000, -10000), new Vec2(1000000, 10000)));
    }
    public void addObject(GameObject o) {
        gameObjList.add(o);
    }
    public String toString() {
        return player.toString();
    }
    public Vec2 getPosition() {
        return player.position;
    }
    public int getCamCollision(int x, int y) {
        for(GameObject gameObj : gameObjList) {
            if(gameObj.camCollisionCheck(x, y)) {
                if(gameObj.isDeathPlane) return 2;
                return 1;
            }
        }
        if(player.camCollisionCheck(x, y)) return 3;
        return 0;
    }
    public boolean doTick(boolean[] input) {
        player.handleVelocity(input);
        player.handlePosition();
        for(GameObject gameObj : gameObjList) {
            if(gameObj.collisionCheck(player.position)) {
                if(gameObj.isDeathPlane) {
                    player.state = 2;
                    return false;
                }
                Vec2 displacement = gameObj.getDisplacement(player.position);
                if(displacement.y == 0) {
                    player.position.x += displacement.x;
                    player.velocity.x = 0;
                }
                else {
                    player.position.y += displacement.y;
                    player.velocity.y = 0;
                    if(displacement.y > 0) player.state = 1;
                }
            }
        }
        return true;
    }
}
