use std::any::TypeId;

use crate::{
    collective::CommunicatorCollectives,
    point_to_point::{Destination, Source},
    raw::AsRaw,
    topology::{Communicator, SimpleCommunicator},
    traits::Equivalence,
};

/// A typed communicator for MPI operations with data type T.
pub struct TypedCommunicator<'a, T>
where
    T: Equivalence,
{
    communicator: &'a SimpleCommunicator, // Reference to avoid ownership issues
    phantom: std::marker::PhantomData<T>,
}

#[derive(Eq, PartialEq, Equivalence, Debug, Clone, Default)]
#[mpi(crate = "crate")]
struct MyTypeId(u64, u64);

impl<'a, T> TypedCommunicator<'a, T>
where
    T: Equivalence + 'static,
{
    /// Creates a new `TypedCommunicator` over type `T`.
    pub fn new(communicator: &'a SimpleCommunicator) -> Self {
        // Validate datatype during construction
        let rank = communicator.rank();
        let size = communicator.size();

        let local_type = TypeId::of::<T>();
        let local_type: MyTypeId = unsafe { std::mem::transmute(local_type) };

        // Collect datatype info across ranks
        let mut all_types = vec![MyTypeId::default(); size as usize];
        communicator.all_gather_into(&local_type, &mut all_types);

        // Check congruence
        let is_congruent = all_types.iter().all(|dt| dt == &local_type);
        if !is_congruent {
            panic!(
                "Rank {}: Datatype mismatch detected among ranks: {:?}",
                rank, all_types
            );
        } else if rank == 0 {
            println!(
                "Rank {}: Datatypes validated successfully: {:?}",
                rank, all_types
            );
        }

        TypedCommunicator {
            communicator,
            phantom: std::marker::PhantomData,
        }
    }

    /// Sends a single value to the specified destination.

    pub fn send_value(&self, data: &T, destination: i32, _tag: i32) {
        // Type-checking for `send_value`
        if T::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `send_value`: Cannot send data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<T>(),
                std::any::type_name::<T>()
            );
        }

        self.communicator.process_at_rank(destination).send(data);
    }

    /// Sends a slice of values to the specified destination.
    pub fn send_slice<U>(&self, data: &[U], destination: i32, _tag: i32)
    where
        U: Equivalence,
    {
        // Type-checking for `send_slice`
        if U::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `send_slice`: Cannot send data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<U>(),
                std::any::type_name::<T>()
            );
        }

        self.communicator.process_at_rank(destination).send(data);
    }

    /// Receives a single value from the specified source.
    pub fn receive_value(&self, buffer: &mut T, source: i32, _tag: i32) {
        // Type-checking for `receive_value`
        if T::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `receive_value`: Cannot receive data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<T>(),
                std::any::type_name::<T>()
            );
        }

        self.communicator
            .process_at_rank(source)
            .receive_into(buffer);
    }

    /// Receives a slice of values from the specified source.
    pub fn receive_slice<U>(&self, buffer: &mut [U], source: i32, _tag: i32)
    where
        U: Equivalence,
    {
        // Type-checking for `receive_slice`
        if U::equivalent_datatype().as_raw() != T::equivalent_datatype().as_raw() {
            panic!(
                "Type mismatch in `receive_slice`: Cannot receive data of type {:?} with a communicator for type {:?}",
                std::any::type_name::<U>(),
                std::any::type_name::<T>()
            );
        }

        self.communicator
            .process_at_rank(source)
            .receive_into(buffer);
    }
}
