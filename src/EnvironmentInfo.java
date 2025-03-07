import game;
import graphics;

public class EnvironmentInfo {
    int[][] objects;
    GameHandler game;
    DisplayHandler display;
    public EnvironmentInfo() {
        File f = new File("levels/the_tower.txt");
        objects = TextFileConverter.txtToIntArray(f);
        reset();
    }
    public void reset() {
        game = new GameHandler(1000, 11000);
        game.loadObjects(objects);
        display = new DisplayHandler(game, 60, 30);
    }
    public int[][] getState() {
        return display.getPixelBuffer();
    }
}