package graphics;

import game.GameHandler;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import javax.swing.JPanel;

public class GamePanel extends JPanel {
    public static final int WIDTH = 800;
    public static final int HEIGHT = 800;
    public static final int FPS = 60;
    public static final int X_RESOLUTION = 144;
    public static final int Y_RESOLUTION = 144;
    private GameHandler game;
    private DisplayHandler display;
    private KeyHandler inputGetter;

    public GamePanel() {
        super();
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setLayout(null);
        game = new GameHandler();
        display = new DisplayHandler(game, X_RESOLUTION, Y_RESOLUTION);
        inputGetter = new KeyHandler();
        addKeyListener(inputGetter);
        setFocusable(true);
    }

    public void start() {
        final double timeBetweenFrames = 1000000000/FPS;
		double delta = 0;
		double lastUpdateTime = System.nanoTime();
		double currentTime = System.nanoTime();
		while(gameThread != null) {
			currentTime = System.nanoTime();
			delta += (currentTime - lastUpdateTime) / timeBetweenFrames;
			lastUpdateTime = currentTime;
			if(delta >= 1) {
				requestFocusInWindow();
				update();
				repaint();
				delta--;
			}
		}
    }
}