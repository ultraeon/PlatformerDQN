import java.util.ArrayList;

public class GameHandler {
    private ArrayList<GameObject> gameObjList;
    private Player player;
    public GameHandler() {
        player = new Player();
        gameObjList = new ArrayList<GameObject>();
        gameObjList.add(new GameObject(new Vec2(-500000, -10000), new Vec2(1000000, 10000)));
    }
    public String toString() {
        return player.toString();
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
