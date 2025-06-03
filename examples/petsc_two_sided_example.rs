use mpi::initialize;
use mpi::petsc_two_sided::{DynamicBuffer, build_two_sided_ibarrier};
use mpi::traits::*;

fn main() {
    let universe = initialize().unwrap();
    let world = universe.world();
    
    let rank = world.rank();
    let size = world.size();
    
    println!("Process {} of {} starting", rank, size);
    
    // Example: Each process sends data to the next process (circular)
    let target_rank = (rank + 1) % size;
    let to_ranks = vec![target_rank];
    
    // Send different data types based on rank
    let to_data = match rank % 4 {
        0 => vec![DynamicBuffer::F32(vec![1.0, 2.0, 3.0])],
        1 => vec![DynamicBuffer::F64(vec![4.0, 5.0])],
        2 => vec![DynamicBuffer::I32(vec![6, 7, 8, 9])],
        _ => vec![DynamicBuffer::I64(vec![10, 11])],
    };
    
    println!("Rank {} sending to rank {}", rank, target_rank);
    
    // Perform two-sided communication
    match build_two_sided_ibarrier(&world, &to_ranks, &to_data) {
        Ok(result) => {
            println!("Rank {} received data from {} processes:", rank, result.from_ranks.len());
            for (i, from_rank) in result.from_ranks.iter().enumerate() {
                println!("  From rank {}: {:?}", from_rank, result.from_data[i]);
            }
        },
        Err(e) => {
            eprintln!("Error on rank {}: {}", rank, e);
        }
    }
    
    println!("Process {} finished", rank);
}