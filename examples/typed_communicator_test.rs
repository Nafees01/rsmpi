use mpi::{traits::*, typed_communicator::TypedCommunicator};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world_ref = &world; // Use a reference instead of ownership
                            //let typed_comm:TypedCommunicator<i32>  = TypedCommunicator::new(world_ref);

    if world.rank() == 0 {
        let typed_comm: TypedCommunicator<f32> = TypedCommunicator::new(world_ref);

        // Example 1: Send a single value
        let single_value: f32 = 42.5;
        typed_comm.send(&single_value, 1, 0);
        println!("Rank 0 sent single value: {}", single_value);

        // Example 2: Send an array
        let array = [1.0_f32, 2.0, 3.0];
        typed_comm.send(&array, 1, 1);
        println!("Rank 0 sent array: {:?}", array);

        let x = [[1.0_f32; 2]; 3];
        //let flattened = x.concat();
        typed_comm.send(&x, 1, 0);

        println!("Rank 0 sent: {:?}", &x);

        //  let x = [[1.0_f32, 2.0]; 3];

        //  // Send the 2D array directly
        //  typed_comm.send(&x, 1, 0);
        //  println!("Rank 0 sent 2D array: {:?}", x);

        // Send a single value
        // let single_value: f32 = 42.0;
        // typed_comm.send(&[single_value], 1, 0);
        // println!("Rank 0 sent single value: {}", single_value);

        // let array = [1.0_f32, 2.0, 3.0, 4.0];
        // typed_comm.send(&array, 1, 0);
        // println!("Rank 0 sent slice: {:?}", array);
    } else if world.rank() == 1 {
        let typed_comm: TypedCommunicator<f32> = TypedCommunicator::new(world_ref);

        // Example 1: Receive a single value
        let mut single_value: f32 = 0.0;
        typed_comm.receive(&mut single_value, 0, 0);
        println!("Rank 1 received single value: {}", single_value);

        // Example 2: Receive an array
        let mut array = [0.0_f32; 3];
        typed_comm.receive(&mut array, 0, 1);
        println!("Rank 1 received array: {:?}", array);

        let mut y = [[0.0_f32; 3]; 2];
        typed_comm.receive(&mut y, 0, 0);

        println!("Rank 1 received: {:?}", y);

        // Receive a single value
        // let mut single_value = [0.0f32];
        // typed_comm.receive(&mut single_value, 0, 0);
        // println!("Rank 1 received single value: {}", single_value[0]);

        // // Receive a slice
        // let mut buffer = [0.0_f32; 4];
        // typed_comm.receive(&mut buffer, 0, 0);
        // println!("Rank 1 received slice: {:?}", buffer);
    }
}
