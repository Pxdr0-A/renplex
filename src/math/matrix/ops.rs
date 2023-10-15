// std
use std::ops::{Add, Sub};

// local
use super::Matrix;


// Generic addition for Matrix
impl<T> Add for Matrix<T> where 
    T: Add<Output=T>,
    T: Copy {
    
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self {
        assert!(
            self.shape == rhs.shape,
            "Matrix shapes do not match in Addition."
        );

        let mut index = 0;
        let result: Vec<T> = self.body
            .into_iter()
            .map(|x| { 
                index += 1;

                x + rhs.body[index-1]
             })
             .collect();
        
        Matrix { 
            body: result, 
            shape: self.shape, 
            capacity: self.capacity 
        }
    }
}

// Generic addition for Matrix
impl<T> Sub for Matrix<T> where 
    T: Sub<Output=T>,
    T: Copy {
    
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self {
        assert!(
            self.shape == rhs.shape,
            "Matrix shapes do not match in Addition."
        );

        let mut index = 0;
        let result: Vec<T> = self.body
            .into_iter()
            .map(|x| { 
                index += 1;

                x - rhs.body[index-1]
             })
             .collect();
        
        Matrix { 
            body: result, 
            shape: self.shape, 
            capacity: self.capacity 
        }
    }
}