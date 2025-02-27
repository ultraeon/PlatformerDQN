public class Main {
    public static void main(String[] args) throws InterruptedException {
        GameHandler game = new GameHandler();
        game.addObject(new GameObject(new Vec2(5000, 3000), new Vec2(30000, 3000)));
        GameObject o = new GameObject(new Vec2(1000, 3000), new Vec2(4000, 3000));
        o.isDeathPlane = true;
        game.addObject(o);
        DisplayHandler d = new DisplayHandler(game, 60, 30);
        for(int i = 0; i < 100; i++) {
            System.out.println(d);
            Thread.sleep(100);
            game.doTick(new boolean[]{true, false, false});
        }
        for(int i = 0; i < 200; i++) {
            System.out.println(d);
            Thread.sleep(100);
            game.doTick(new boolean[]{false, false, false});
        }
    }
}
