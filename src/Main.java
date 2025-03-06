import game.*;
import graphics.*;
import java.io.File;
import java.io.FileNotFoundException;

public class Main {
    public static void main(String[] args) throws InterruptedException, FileNotFoundException {
        GameHandler game = new GameHandler();
        DisplayHandler d = new DisplayHandler(game, 60, 30);
        File f = new File("levels/test.txt");
        int[][] allObjects = TextFileConverter.txtToIntArray(f);
        game.loadObjects(allObjects);
        for(int i = 0; i < 95; i++) {
            System.out.println(d);
            Thread.sleep(17);
            game.doTick(6);
        }
        for(int i = 0; i < 450; i++) {
            System.out.println(d);
            Thread.sleep(17);
            game.doTick(5);
        }
    }
}
