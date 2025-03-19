use std::any::TypeId;


use crate::{
    collective::{CommunicatorCollectives, SystemOperation, Root},
    point_to_point::{Destination, Source},
    topology::{Communicator, SimpleCommunicator},
    traits::{Equivalence, Buffer, BufferMut},
};


/// A typed communicator for MPI operations with data type T.
pub struct TypedCommunicator<'a, T>
where
    T: 'static,
{
    communicator: &'a SimpleCommunicator, // Reference to avoid ownership issues
    phantom: std::marker::PhantomData<T>,
}

#[derive(Eq, PartialEq, Equivalence, Debug, Clone, Default, PartialOrd, Ord)]
#[mpi(crate = "crate")]
struct MyTypeId(u64, u64);

impl<'a, T> TypedCommunicator<'a, T>
where
    T: 'static,

{
    /// Creates a new `TypedCommunicator` over type `T`.
    pub fn new(communicator: &'a SimpleCommunicator) -> Self {
        // Validate datatype during construction
        let rank = communicator.rank();
        let size = communicator.size();

        let local_type = TypeId::of::<T>();
        let local_type: MyTypeId = unsafe { std::mem::transmute(local_type) };

        // // Collect datatype info across ranks
       // let mut all_types = vec![MyTypeId::default(); size as usize];
        // communicator.all_gather_into(&local_type, &mut all_types);

        // // Check congruence
        // let is_congruent = all_types.iter().all(|dt| dt == &local_type);
        // if !is_congruent {
        //     panic!(
        //         "Rank {}: Datatype mismatch detected among ranks: {:?}",
        //         rank, all_types
        //     );
        // } else if rank == 0 {
        //     println!(
        //         "Rank {}: Datatypes validated successfully: {:?}",
        //         rank, all_types
        //     );
        // }

        
        // Fix the unused variable warning
        let _size = communicator.size();

        // Create a boolean flag instead of using max/min directly on MyTypeId
        let is_same_type = true; // Assume true initially
        let mut result = true;

        // Use all_reduce with logical AND operation
        communicator.all_reduce_into(&is_same_type, &mut result, SystemOperation::logical_and());

        // Before the reduction, broadcast the type from rank 0 and compare locally
        let root_type = if rank == 0 { local_type.clone() } else { MyTypeId::default() };
        let mut bcast_type = root_type;
        communicator.process_at_rank(0).broadcast_into(&mut bcast_type);

        // Each rank compares its type with the broadcast type
        let is_same_type = bcast_type == local_type;
        let mut result = false;

        // Use all_reduce to collect the AND of all comparisons
        communicator.all_reduce_into(&is_same_type, &mut result, SystemOperation::logical_and());

        // Check the result
        if !result {
            panic!(
                "Rank {}: Datatype mismatch detected. Local type: {:?}, Root type: {:?}",
                rank, local_type, bcast_type
            );
        } else if rank == 0 {
            let all_types = vec![local_type.clone(); _size as usize];
            println!(
                "Rank {}: Datatypes validated successfully. Type: {:?}",
                rank, all_types
            );
        }

        TypedCommunicator {
            communicator,
            phantom: std::marker::PhantomData,
        }
    }


    // /// Sends a single value to the specified destination.

    // pub fn send_value<U>(&self, data: &U, destination: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    // {
    //     self.communicator.process_at_rank(destination).send(data);
    // }
    // /// Sends a slice of values to the specified destination.
    // pub fn send_slice<U>(&self, data: &[U], destination: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    // {
    //     self.communicator.process_at_rank(destination).send(data);
    // }
    // /// Receives a single value from the specified source.
    // pub fn receive_value<U>(&self, buffer: &mut U, source: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    // {
    //     self.communicator
    //         .process_at_rank(source)
    //         .receive_into(buffer);
    // }
    // /// Receives a slice of values from the specified source.
    // pub fn receive_slice<U>(&self, buffer: &mut [U], source: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    // {
    //     self.communicator
    //         .process_at_rank(source)
    //         .receive_into(buffer);
    // }

    // /// This function works for both single values and slices.
    // pub fn send<U, D>(&self, data: &D, destination: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    //     D: AsRef<[U]> + Sized,
    // {
    //     let slice = data.as_ref();
    //     self.communicator.process_at_rank(destination).send(slice);
    // }

    // /// This function works for both single values and slices.
    // pub fn receive<U, D>(&self, buffer: &mut D, source: i32, _tag: i32)
    // where
    //     U: Equivalence<Base = T>,
    //     D: AsMut<[U]> + Sized,
    // {
    //     let slice = buffer.as_mut();
    //     self.communicator
    //         .process_at_rank(source)
    //         .receive_into(slice);
    // }


//     /// This function works for both single values and slices.
//     pub fn send<U>(&self, buf: &U, destination: i32, _tag: i32)
//     where
//         U: Buffer,
//     {
//         self.communicator.process_at_rank(destination).send(buf);
//     }

// /// This function works for both single values and slices.
//     pub fn receive<U>(&self, buf: &mut U, source: i32, _tag: i32)
//     where
//         U: BufferMut,
//     {
//         self.communicator.process_at_rank(source).receive_into(buf);
//     }

// /// Sends data to the specified destination.
// pub fn send<U>(&self, buf: &U, destination: i32, _tag: i32)
// where
//     U: Buffer + Equivalence<Base = T>, // Ensure the buffer's base type matches T
// {
//     self.communicator.process_at_rank(destination).send(buf);
// }

// /// Receives data from the specified source.
// pub fn receive<U>(&self, buf: &mut U, source: i32, _tag: i32)
// where
//     U: BufferMut + Equivalence<Base = T>, // Ensure the buffer's base type matches T
// {
//     self.communicator.process_at_rank(source).receive_into(buf);
// }

// /// Sends data to the specified destination.
// pub fn send(&self, buf: &T, destination: i32, _tag: i32)
// where
//     T: Buffer + Equivalence, // Ensure `T` can be used in MPI communication
// {
//     self.communicator.process_at_rank(destination).send(buf);
// }

// /// Receives data from the specified source.
// pub fn receive(&self, buf: &mut T, source: i32, _tag: i32)
// where
//     T: BufferMut + Equivalence, // Ensure `T` can be used in MPI communication
// {
//     self.communicator.process_at_rank(source).receive_into(buf);
// }

//  /// Pathao
//  pub fn send(&self, buf: &T, destination: i32, _tag: i32)
//  where
//      T: Buffer, // Ensures that only `T` can be sent
//  {
//      self.communicator.process_at_rank(destination).send(buf);
//  }

//  /// Grohon
//  pub fn receive(&self, buf: &mut T, source: i32, _tag: i32)
//  where
//      T: BufferMut, // Ensures that only `T` can be received
//  {
//      self.communicator.process_at_rank(source).receive_into(buf);
//  }

//  /// Unified send function that works for both single values and slices
//  pub fn send<B>(&self, data: &B, destination: i32, _tag: i32)
//  where
//      B: ?Sized + Buffer,
//      T: Buffer,
//  {
//      self.communicator.process_at_rank(destination).send(data);
//  }

//  /// Unified receive function that works for both single values and slices
//  pub fn receive<B>(&self, buffer: &mut B, source: i32, _tag: i32)
//  where
//      B: ?Sized + BufferMut,
//      T: BufferMut,
//  {
//      self.communicator.process_at_rank(source).receive_into(buffer);
//  }

/// Sends data to the specified destination. Works with both single values and slices.
/// 
/// # Arguments
/// * `data` - The data to send, can be a single value or slice
/// * `destination` - Rank of the destination process
/// * `_tag` - Message tag
pub fn send(&self, data: &[T], destination: i32, _tag: i32)
where
    T: Buffer + Equivalence,
{
    self.communicator.process_at_rank(destination).send(data);
}

/// Receives data from the specified source. Works with both single values and slices.
/// 
/// # Arguments
/// * `buffer` - Buffer to receive the data into, can be a single value or slice
/// * `source` - Rank of the source process
/// * `_tag` - Message tag
pub fn receive(&self, buffer: &mut [T], source: i32, _tag: i32)
where
    T: BufferMut + Equivalence,
{
    self.communicator.process_at_rank(source).receive_into(buffer);
}

}

