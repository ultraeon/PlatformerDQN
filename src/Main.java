public class Main {
    public static void main(String[] args) throws InterruptedException {
        GameHandler game = new GameHandler();
        game.addObject(new GameObject(new Vec2(5000, 0), new Vec2(100000, 4000)));
        DisplayHandler d = new DisplayHandler(game, 40, 40);
        for(int i = 0; i < 1000; i++) {
            System.out.println(d);
            Thread.sleep(100);
            game.doTick(new boolean[]{true, false, true});
        }
    }
}
