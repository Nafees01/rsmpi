//! PETSc-style two-sided communication module for MPI operations.

use crate::traits::*;
use crate::topology::{Communicator, Rank};
use crate::point_to_point::Source;
use crate::collective::CommunicatorCollectives;
use std::collections::VecDeque;
use std::time::Duration;

/// Dynamic buffer that can hold different numeric types for MPI communication.
#[derive(Debug, Clone)]
pub enum DynamicBuffer {
    /// 32-bit floating point vector
    F32(Vec<f32>),
    /// 64-bit floating point vector
    F64(Vec<f64>),
    /// 32-bit signed integer vector
    I32(Vec<i32>),
    /// 64-bit signed integer vector
    I64(Vec<i64>),
}

impl DynamicBuffer {
    /// Returns the length of the buffer
    pub fn len(&self) -> usize {
        match self {
            DynamicBuffer::F32(v) => v.len(),
            DynamicBuffer::F64(v) => v.len(),
            DynamicBuffer::I32(v) => v.len(),
            DynamicBuffer::I64(v) => v.len(),
        }
    }

    /// Returns a type identifier for the buffer contents
    pub fn type_id(&self) -> u8 {
        match self {
            DynamicBuffer::F32(_) => 0,
            DynamicBuffer::F64(_) => 1,
            DynamicBuffer::I32(_) => 2,
            DynamicBuffer::I64(_) => 3,
        }
    }
}

/// Message header containing metadata about the data being sent
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageHeader {
    /// Type identifier for the data
    pub data_type: u8,
    /// Number of elements in the data
    pub count: u64,
}

impl MessageHeader {
    /// Converts the header to a byte array for transmission
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut buffer = [0u8; 9];
        buffer[0] = self.data_type;
        buffer[1..].copy_from_slice(&self.count.to_le_bytes());
        buffer
    }

    /// Creates a header from a byte array
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 9 {
            return Err(format!("Invalid header length: expected 9, got {}", bytes.len()));
        }
        let data_type = bytes[0];
        let mut count_bytes = [0u8; 8];
        count_bytes.copy_from_slice(&bytes[1..9]);
        let count = u64::from_le_bytes(count_bytes);
        Ok(MessageHeader { data_type, count })
    }
}

/// Result of a two-sided communication operation
pub struct TwoSidedResult {
    /// Ranks that sent data to this process
    pub from_ranks: Vec<Rank>,
    /// Data received from other processes
    pub from_data: Vec<DynamicBuffer>,
}

struct SegmentedBuffer<T> {
    data: VecDeque<T>,
    #[allow(dead_code)]
    chunk_size: usize,
}

impl<T> SegmentedBuffer<T> {
    fn new(chunk_size: usize) -> Self {
        Self {
            data: VecDeque::new(),
            chunk_size,
        }
    }

    fn push(&mut self, item: T) {
        self.data.push_back(item);
    }

    fn extract_all(self) -> Vec<T> {
        self.data.into()
    }
}

/// PETSc-style two-sided communication using non-blocking sends and probe/barrier for termination
pub fn build_two_sided_ibarrier<C: Communicator>(
    comm: &C,
    to_ranks: &[Rank],
    to_data: &[DynamicBuffer],
) -> Result<TwoSidedResult, Box<dyn std::error::Error>> {

    let tag_header = 100;
    let tag_data = 101;

    assert_eq!(to_ranks.len(), to_data.len(), "Mismatched ranks and data");

    let mut seg_ranks = SegmentedBuffer::new(4);
    let mut seg_data = SegmentedBuffer::new(4);

    // Send all messages using blocking sends to avoid scope issues
    for (i, &target_rank) in to_ranks.iter().enumerate() {
        let data = &to_data[i];
        let header = MessageHeader {
            data_type: data.type_id(),
            count: data.len() as u64,
        };

        let header_bytes = header.to_bytes();
        comm.process_at_rank(target_rank)
            .send_with_tag(&header_bytes, tag_header);

        match data {
            DynamicBuffer::F32(vec) => {
                comm.process_at_rank(target_rank)
                    .send_with_tag(vec.as_slice(), tag_data);
            },
            DynamicBuffer::F64(vec) => {
                comm.process_at_rank(target_rank)
                    .send_with_tag(vec.as_slice(), tag_data);
            },
            DynamicBuffer::I32(vec) => {
                comm.process_at_rank(target_rank)
                    .send_with_tag(vec.as_slice(), tag_data);
            },
            DynamicBuffer::I64(vec) => {
                comm.process_at_rank(target_rank)
                    .send_with_tag(vec.as_slice(), tag_data);
            },
        }
    }

    let mut barrier_request = Some(comm.immediate_barrier());
    let start_time = std::time::Instant::now();
    let timeout = Duration::from_secs(30);

    loop {
        if start_time.elapsed() > timeout {
            return Err("Communication timeout - possible deadlock".into());
        }

        let barrier_complete = if let Some(barrier) = barrier_request.take() {
            match barrier.test() {
                Ok(_) => true,
                Err(req) => {
                    barrier_request = Some(req);
                    false
                }
            }
        } else {
            false
        };

        // Try to probe for incoming messages (non-blocking)
        if let Some(status) = comm.any_process().immediate_probe_with_tag(tag_header) {
            // Receive header
            let (header_data, _): (Vec<u8>, _) = comm.process_at_rank(status.source_rank())
                .receive_vec_with_tag(tag_header);
            
            let header = MessageHeader::from_bytes(&header_data)?;

            let received_data = match header.data_type {
                0 => {
                    let (data, _) = comm.process_at_rank(status.source_rank())
                        .receive_vec_with_tag(tag_data);
                    DynamicBuffer::F32(data)
                },
                1 => {
                    let (data, _) = comm.process_at_rank(status.source_rank())
                        .receive_vec_with_tag(tag_data);
                    DynamicBuffer::F64(data)
                },
                2 => {
                    let (data, _) = comm.process_at_rank(status.source_rank())
                        .receive_vec_with_tag(tag_data);
                    DynamicBuffer::I32(data)
                },
                3 => {
                    let (data, _) = comm.process_at_rank(status.source_rank())
                        .receive_vec_with_tag(tag_data);
                    DynamicBuffer::I64(data)
                },
                _ => return Err("Unknown data type".into()),
            };

            seg_ranks.push(status.source_rank());
            seg_data.push(received_data);
        } else if barrier_complete {
            break;
        }

        std::thread::sleep(Duration::from_millis(1));
    }

    Ok(TwoSidedResult {
        from_ranks: seg_ranks.extract_all(),
        from_data: seg_data.extract_all(),
    })
}        
