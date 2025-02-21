public class Vec2 {
    int x;
    int y;
    public Vec2(int x, int y) {
        this.x = x;
        this.y = y;
    }
    public Vec2 add(Vec2 other) {
        return new Vec2(x+other.x, y+other.y);
    }
    public Vec2 subtract(Vec2 other) {
        return new Vec2(x-other.x, y-other.y);
    }
    public Vec2 hadamardMultiply(Vec2 other) {
        return new Vec2(x*other.x, y*other.y);
    }
    public int dotMultiply(Vec2 other) {
        return x*other.x+y*other.y;
    }
}
