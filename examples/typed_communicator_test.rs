use mpi::{traits::*, typed_communicator::TypedCommunicator};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world_ref = &world; // Use a reference instead of ownership
                            //let typed_comm: TypedCommunicator<f32> = TypedCommunicator::new(world_ref);

    if world.rank() == 0 {
        let typed_comm = TypedCommunicator::new(world_ref);
        //     let data: f32 = 42.5;
        //     typed_comm.send_value(&data, 1, 0);
        //    println!("Rank 0 sent {}", data);
        let x = [[1.0_f32; 2]; 3];
        let flattened = x.concat();
        typed_comm.send_slice(&flattened, 1, 0);
        println!("Rank 0 sent: {:?}", flattened);
    } else if world.rank() == 1 {
        let typed_comm = TypedCommunicator::new(world_ref);
        //     let mut buffer: f32 = 0.0;
        //    typed_comm.receive_value(&mut buffer, 0, 0);
        //    println!("Rank 1 received {}", buffer);
        let mut buffer = vec![0.0_f32; 6];
        typed_comm.receive_slice(&mut buffer, 0, 0);

        let y: [[f32; 2]; 3] = [
            [buffer[0], buffer[1]],
            [buffer[2], buffer[3]],
            [buffer[4], buffer[5]],
        ];
        println!("Rank 1 received: {:?}", y);
    }
}
