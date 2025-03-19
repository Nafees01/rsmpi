use mpi::{traits::*, typed_communicator::TypedCommunicator};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world_ref = &world; // Use a reference instead of ownership
    //let typed_comm:TypedCommunicator<i32>  = TypedCommunicator::new(world_ref);

    if world.rank() == 0 {
      let typed_comm: TypedCommunicator<f32>  = TypedCommunicator::new(world_ref);
        //     let data: i32 = 42;
        //     typed_comm.send(&[data], 1, 0);
        //    println!("Rank 0 sent {}", data);



        let x = [[1.0_f32; 2]; 3];
        let flattened = x.concat();
        typed_comm.send(&flattened, 1, 0);
    

         println!("Rank 0 sent: {:?}", flattened);

    
        // Send a single value
        // let single_value: f32 = 42.0;
        // typed_comm.send(&[single_value], 1, 0);
        // println!("Rank 0 sent single value: {}", single_value);


        // let array = [1.0_f32, 2.0, 3.0, 4.0];
        // typed_comm.send(&array, 1, 0);
        // println!("Rank 0 sent slice: {:?}", array);
    } else if world.rank() == 1 {

       let typed_comm: TypedCommunicator<f32> = TypedCommunicator::new(world_ref);
        // let mut buffer = [0.0_f32; 1];
        // typed_comm.receive(&mut buffer, 0, 0);  // Pass the array directly
        // println!("Rank 1 received {}", buffer[0]);




        let mut buffer = vec![0.0_f32; 6];
        typed_comm.receive(&mut buffer, 0, 0);


        let y: [[f32; 3]; 2] = [
            [buffer[0], buffer[1], buffer[2]],
            [buffer[3], buffer[4], buffer[5]],
        ];

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
