import game.*;
import graphics.*;
import java.io.File;
import java.io.FileNotFoundException;

public class Main {
    public static void main(String[] args) throws InterruptedException, FileNotFoundException {
        GameHandler game = new GameHandler(1000, 11000);
        DisplayHandler d = new DisplayHandler(game, 60, 30);
        File f = new File("levels/the_tower.txt");
        int[][] allObjects = TextFileConverter.txtToIntArray(f);
        game.loadObjects(allObjects);
        for(int i = 0; i < 250; i++) {
            System.out.println(d);
            Thread.sleep(17);
            if(!game.doTick(0b101)) break;
        }
        for(int i = 0; i < 100; i++) {
            System.out.println(d);
            Thread.sleep(17);
            if(!game.doTick(0b000)) break;
        }
        for(int i = 0; i < 100000; i++) {
            System.out.println(d);
            Thread.sleep(17);
            if(!game.doTick(0b101)) break;
        }
    }
}
