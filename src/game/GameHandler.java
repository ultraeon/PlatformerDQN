import java.util.ArrayList;

// class to handle internals of the game
public class GameHandler {
    private ArrayList<GameObject> gameObjList;
    private Player player;
    public GameHandler() {
        player = new Player();
        gameObjList = new ArrayList<GameObject>();
        gameObjList.add(new GameObject(new Vec2(-500000, -10000), new Vec2(1000000, 10000)));
        gameObjList.get(0).isVisible = true;
        gameObjList.get(0).isTangible = true;
    }
    // adds an object as is
    public void addObject(GameObject o) {
        gameObjList.add(o);
    }
    // adds objects from an int array
    // each row is an object of form [x, y, width, height, boolean information]
    public void loadObjects(int[][] objList) {
        GameObject o;
        for(int[] obj : objList) {
            o = new GameObject(new Vec2(obj[0], obj[1]), new Vec2(obj[2], obj[3]));
            if((obj[4] & 1) == 1) o.isVisible = true;
            if(((obj[4] >> 1) & 1) == 1) o.isTangible = true;
            if(((obj[4] >> 2) & 1) == 1) o.isDeathPlane = true;
            gameObjList.add(o);
        }
    }
    public String toString() {
        return player.toString();
    }
    public Vec2 getPosition() {
        return player.position;
    }
    public int getCamCollision(int x, int y) {
        for(GameObject gameObj : gameObjList) {
            if(!gameObj.isVisible) continue;
            if(gameObj.camCollisionCheck(x, y)) {
                if(gameObj.isDeathPlane) return 2;
                return 1;
            }
        }
        if(player.camCollisionCheck(x, y)) return 3;
        return 0;
    }
    public boolean doTick(boolean[] input) {
        if(player.state == 2) return false;
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
                else if(displacement.x == 0){
                    player.position.y += displacement.y;
                    player.velocity.y = 0;
                    if(displacement.y > 0) player.state = 1;
                }
            }
        }
        return true;
    }
}
