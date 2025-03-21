import game.*;
import graphics.*;
import java.io.File;
import java.io.FileNotFoundException;

public class EnvironmentInfo {
    int[][] objects;
    GameHandler game;
    DisplayHandler display;
    public EnvironmentInfo() throws FileNotFoundException {
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
    public int doTick(int input) {
        int p1 = game.getPosition().x;
        if(!game.doTick(input)) return -50;
        return game.getPosition().x-p1;
    }
}