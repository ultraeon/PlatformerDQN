public class Main {
    public static void main(String[] args) throws InterruptedException {
        GameHandler game = new GameHandler();
        DisplayHandler d = new DisplayHandler(game, 60, 30);
        int[][] test = new int[30][5];
        for(int i = 0; i < 30; i++) {
            test[i][0] = (int)(Math.random()*100000);
            test[i][1] = (int)(Math.random()*4000);
            test[i][2] = (int)(Math.random()*2000)+1000;
            test[i][3] = (int)(Math.random()*2000)+1000;
            test[i][4] = 3;
        }
        game.loadObjects(test);
        for(int i = 0; i < 1000; i++) {
            System.out.println(d);
            Thread.sleep(100);
            game.doTick(new boolean[]{true, false, true});
        }
    }
}
