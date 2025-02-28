package input;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class KeyHandler implements KeyListener {
	public static boolean rightPressed;
	public static boolean leftPressed;
	public static boolean upPressed;
	
	@Override
	public void keyPressed(KeyEvent e) {
		int kCode = e.getKeyCode();
		if(kCode == KeyEvent.VK_RIGHT || kCode == KeyEvent.VK_D)
			rightPressed = true;
		if(kCode == KeyEvent.VK_LEFT || kCode == KeyEvent.VK_A)
			leftPressed = true;
		if(kCode == KeyEvent.VK_UP || kCode == KeyEvent.VK_W)
			upPressed = true;
	}
	@Override
	public void keyReleased(KeyEvent e) {
		int kCode = e.getKeyCode();
		if(kCode == KeyEvent.VK_RIGHT || kCode == KeyEvent.VK_D)
			rightPressed = false;
		if(kCode == KeyEvent.VK_LEFT || kCode == KeyEvent.VK_A)
			leftPressed = false;
		if(kCode == KeyEvent.VK_UP || kCode == KeyEvent.VK_W)
			upPressed = false;
	}
	@Override
	public void keyTyped(KeyEvent e) {
	}
}