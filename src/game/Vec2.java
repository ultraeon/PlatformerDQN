package game;
// class for a vector of length 2 -> <x, y>
public class Vec2 {
    public int x;
    public int y;
    public Vec2(int x, int y) {
        this.x = x;
        this.y = y;
    }
    // prints as vector
    public String toString() {
        return "<"+x+", "+y+">";
    }
    // elementwise addition
    public Vec2 add(Vec2 other) {
        return new Vec2(x+other.x, y+other.y);
    }
    // elementwise subtraction
    public Vec2 subtract(Vec2 other) {
        return new Vec2(x-other.x, y-other.y);
    }
    // scalar multiplication a*<x, y> = <ax, ay>
    public Vec2 sMultiply(int m) {
        return new Vec2(x*m, y*m);
    }
    // hadamard multiplication <a, b>*<c, d> = <ac, bd>
    public Vec2 hMultiply(Vec2 other) {
        return new Vec2(x*other.x, y*other.y);
    }
    // dot product calculation <a, b>*<c, d> = ac+bd
    public int dMultiply(Vec2 other) {
        return x*other.x+y*other.y;
    }
}
