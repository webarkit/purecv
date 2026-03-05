/*
 *  types_test.rs (Internal test module)
 */

#[cfg(test)]
mod tests {
    use crate::core::types::*;
    use crate::core::utils::*;

    #[test]
    fn test_point_add() {
        let p1 = Point2i::new(10, 20);
        let p2 = Point2i::new(5, 5);
        let p3 = p1 + p2;
        assert_eq!(p3.x, 15);
        assert_eq!(p3.y, 25);
    }

    #[test]
    fn test_size_area() {
        let sz = Size2i::new(100, 50);
        assert_eq!(sz.area(), 5000);
    }

    #[test]
    fn test_rect_tl_br() {
        let r = Rect2i::new(10, 10, 100, 50);
        assert_eq!(r.tl(), Point2i::new(10, 10));
        assert_eq!(r.br(), Point2i::new(110, 60));
    }

    #[test]
    fn test_range() {
        let r = Range::new(10, 20);
        assert_eq!(r.size(), 10);
        assert!(!r.empty());
        
        let r_all = Range::all();
        assert_eq!(r_all.start, i32::MIN);
    }

    #[test]
    fn test_scalar() {
        let s = Scalar::<u8>::all(255);
        assert_eq!(s.values, [255, 255, 255, 255]);
    }
}
