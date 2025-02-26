public class DisplayHandler {
    GameHandler game;
    int resolution;
    public static final int CAMERA_RANGE = 10000;
    public DisplayHandler(GameHandler g, int res) {
        game = g;
        resolution = res;
    }
    public int[][] getPixelBuffer() {
        int xPosition = game.getPosition().x;
        int yPosition = game.getPosition().y;
        int[][] pBuffer = new int[resolution][resolution];
        int camX, camY;
        for(int i = 0; i < resolution; i++) {
            for(int j = 0; j < resolution; j++) {
                camX = i*CAMERA_RANGE/resolution+xPosition-CAMERA_RANGE/2;
                camY = j*CAMERA_RANGE/resolution+yPosition-CAMERA_RANGE/2;
                pBuffer[i][j] = game.getCamCollision(camX, camY);
                System.out.println(camX + " " + camY);
            }
        }
        return pBuffer;
    }
    public String toString() {
        int[][] pBuffer = getPixelBuffer();
        String s = "";
        for(int i = resolution-1; i > -1; i--) {
            for(int j = 0; j < resolution; j++) {
                switch(pBuffer[j][i]) {
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
