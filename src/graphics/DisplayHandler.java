package graphics;
import game.GameHandler;

public class DisplayHandler {
    GameHandler game;
    int xResolution;
    int yResolution;
    public static final int CAMERA_X_RANGE = 14000;
    public static final int CAMERA_Y_RANGE = 10000;
    public DisplayHandler(GameHandler g, int xRes, int yRes) {
        game = g;
        xResolution = xRes;
        yResolution = yRes;
    }
    public int[][] getPixelBuffer() {
        int xPosition = game.getPosition().x+500;
        int yPosition = game.getPosition().y+3000;
        int[][] pBuffer = new int[xResolution][yResolution];
        int camX, camY;
        for(int i = 0; i < xResolution; i++) {
            for(int j = 0; j < yResolution; j++) {
                camX = i*CAMERA_X_RANGE/xResolution+xPosition-CAMERA_X_RANGE/2;
                camY = j*CAMERA_Y_RANGE/yResolution+yPosition-CAMERA_Y_RANGE/2;
                pBuffer[i][j] = game.getCamCollision(camX, camY);
            }
        }
        return pBuffer;
    }
    public String toString() {
        int[][] pBuffer = getPixelBuffer();
        String s = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
        for(int j = yResolution-1; j > -1; j--) {
            for(int i = 0; i < xResolution; i++) {
                switch(pBuffer[i][j]) {
                    case 0: s+=" "; break;
                    case 1: s+="o"; break;
                    case 2: s+="/"; break;
                    case 3: s+="z"; break;
                }
            }
            s += "\n";
        }
        return s;
    }
}
