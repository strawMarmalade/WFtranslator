use std::cmp;
use std::collections::HashMap;

use ndarray::NewAxis;

/// Adjacency matrix.
type AdjMtx = HashMap<usize, Vec<usize>>;

/// This struct represents a graph.
#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    //degree: u64,
    max_degree: usize,
    min_degree: usize,
    avg_degree_frac: (usize, usize),
    max_core: usize,
    adjmtx: AdjMtx,
    is_gstats: bool,
    kcore: Vec<usize>,
    kcore_order: Vec<usize>,
}

impl Graph {
    /// Creates a new graph with disconnected nodes and returns it.
    pub fn new(nodes: usize) -> Graph {
        assert!(nodes != 0, "The number of nodes cannot be zero.");
        let mut adjmtx = AdjMtx::new();
        for n in 1..=nodes {
            adjmtx.insert(n, vec![]);
        }
        Graph {
            adjmtx,
            max_degree: 0,
            min_degree: 0,
            avg_degree_frac: (0, 1),
            max_core: 0,
            is_gstats: false,
            kcore: vec![],
            kcore_order: vec![],
        }
    }

    /// Returns the graph degree.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the degree of a node.
    pub fn degree_of(&self, n: usize) -> usize {
        assert!(
            self.contains_node(n),
            "The given node does not belong to the graph"
        );
        self.adjmtx[&n].len()
    }

    /// Returns the list of nodes of the graph.
    pub fn nodes(&self) -> Vec<usize> {
        let mut nodes: Vec<usize> = vec![];
        for n in self.adjmtx.keys() {
            nodes.push(*n);
        }
        nodes
    }

    /// Returns the list of nodes of the graph ordered by their degree.
    pub fn nodes_ord_by_degree(&self) -> Vec<usize> {
        let mut nds_dgs: Vec<(usize, usize)> = vec![];
        for &n in self.adjmtx.keys() {
            nds_dgs.push((self.degree_of(n), n));
        }
        nds_dgs.sort_by_key(|&w| cmp::Reverse(w));
        nds_dgs.iter().map(|(_, n)| *n).collect()
    }

    /// Returns the list of edges of the graph.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges: Vec<(usize, usize)> = vec![];
        let mut nodes = self.nodes();
        nodes.sort();
        for n in nodes {
            for &an in &self.adjmtx[&n] {
                if !edges.contains(&(an, n)) {
                    edges.push((n, an));
                }
            }
        }
        edges
    }

    /// Returns the adjacency list of a node.
    pub fn adjlst_of(&self, n: usize) -> &Vec<usize> {
        &self.adjmtx[&n]
    }

    /// Inserts a new node in the graph.
    pub fn insert_node(&mut self, n: usize) {
        assert!(
            !self.contains_node(n),
            "The given node already belongs to the graph"
        );
        self.adjmtx.insert(n, vec![]);
    }

    /// Inserts an edge in the graph.
    pub fn insert_edge(&mut self, (a, b): (usize, usize)) {
        if let Some(lst) = self.adjmtx.get_mut(&a) {
            lst.push(b);
            self.max_degree = cmp::max(self.max_degree, lst.len());
        } else {
            panic!("The node {} does not belong to this graph.", a);
        }
        if let Some(lst) = self.adjmtx.get_mut(&b) {
            lst.push(a);
            self.max_degree = cmp::max(self.max_degree, lst.len());
        } else {
            panic!("The node {} does not belong to this graph.", b);
        }
    }

    /// Removes a node from the graph.
    pub fn remove_node(&mut self, n: usize) {
        assert!(
            self.contains_node(n),
            "This graph does not contains the given node."
        );
        self.min_degree = cmp::min(self.min_degree, self.degree_of(n));
        self.adjmtx.remove(&n);
        for (_, v) in self.adjmtx.iter_mut() {
            if let Some(index) = v.iter().position(|x| *x == n) {
                v.remove(index);
            }
        }
    }

    /// Removes an edge from the graph.
    pub fn remove_edge(&mut self, (a, b): (usize, usize)) {
        assert!(
            self.contains_node(a) && self.contains_node(b),
            "This graph does not contains at least one of the given nodes."
        );
        assert!(
            self.adjmtx[&a].contains(&b) && self.adjmtx[&b].contains(&a),
            "The first node is not adjacent to the second one"
        );
        if let Some(v) = self.adjmtx.get_mut(&a) {
            if let Some(index) = v.iter().position(|x| *x == b) {
                v.remove(index);
            }
        }
        if let Some(v) = self.adjmtx.get_mut(&b) {
            if let Some(index) = v.iter().position(|x| *x == a) {
                v.remove(index);
            }
        }
    }

    /// Returns true if the graph contains the node and false otherwise.
    pub fn contains_node(&self, n: usize) -> bool {
        self.adjmtx.contains_key(&n)
    }

    /// Returns true if the graph contains the edge and false otherwise.
    pub fn contains_edge(&self, e: (usize, usize)) -> bool {
        if !self.adjmtx.contains_key(&e.0) {
            return false;
        }
        if !self.adjmtx.contains_key(&e.1) {
            return false;
        }
        if !self.adjmtx[&e.0].contains(&e.1) {
            return false;
        }
        if !self.adjmtx[&e.1].contains(&e.0) {
            return false;
        }
        true
    }

    //Updates the degree values of the graph
    //Must be called to update min degree after removing adding an edge
    //or after removing an edge to update max degree
    pub fn vertex_degrees(&mut self) {
        let orderedNodes = self.nodes_ord_by_degree();
        self.max_degree = self.degree_of(orderedNodes[0]);
        self.min_degree = self.degree_of(orderedNodes[orderedNodes.len() - 1]);
        self.avg_degree_frac = (self.sum_vertex_degrees(), self.adjmtx.len())
    }

    //Sums the degree of every node in the graph.
    pub fn sum_vertex_degrees(&self) -> usize {
        let mut sum_of_deg: usize = 0;
        for &n in self.adjmtx.keys() {
            sum_of_deg += self.degree_of(n);
        }
        sum_of_deg
    }

    /// Returns true if the graph is complete and false otherwise.
    pub fn is_complete(&self) -> bool {
        let nodes = self.nodes();
        for (i, k1) in self.adjmtx.keys().enumerate() {
            // If at least one node has a different number of adjacent nodes
            if i > 0 && self.degree_of(nodes[i]) != self.degree_of(nodes[i - 1]) {
                return false;
            }
            // Check if all nodes are adjacent to each other
            for k2 in self.adjmtx.keys() {
                if k1 == k2 {
                    continue;
                }
                if !self.adjmtx[k1].contains(&k2) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns true if the graph is empty and false otherwise.
    pub fn is_empty(&self) -> bool {
        if self.adjmtx.is_empty() {
            assert!(self.max_degree == 0);
        }
        self.adjmtx.is_empty()
    }

    /// Returns the number of nodes of the graph.
    pub fn nlen(&self) -> usize {
        self.adjmtx.len()
    }

    /// Returns the number of edges of the graph.
    pub fn elen(&self) -> usize {
        let mut sum = 0;
        for adjlst in self.adjmtx.values() {
            sum += adjlst.iter().len();
        }
        sum / 2
    }

    pub fn compute_cores(&mut self) {
        let j: usize = 0;
        let n: usize = self.nlen();
        let i: usize = 0;
        let w: usize = 0;
        let du: usize = 0;
        let pu: usize = 0;
        let pw: usize = 0;

        let nodes = self.nodes();
        let edges = self.edges();

        let mut pos: Vec<usize> = vec![];
        if self.kcore_order.len() > 0 {
            //let tmp: Vec<usize> = vec![0; n];
            self.kcore = vec![0; n];
            self.kcore_order = vec![0; n];
        } else {
            self.kcore_order.resize(n, 0);
            self.kcore.resize(n, 0);
        }

        let mut md: usize = 0;
        let a = self.adjmtx.keys();
        for v in 1..n {
            self.kcore[v] = nodes[v] - nodes[v - 1];
            if self.kcore[v] > md {
                md = self.kcore[v];
            }
        }

        let md_end: usize = md + 1;

        let bin: Vec<usize> = vec![0; md_end];
        for v in 1..n {
            bin[self.kcore[v]] += 1;
        }

        let start: usize = 1;
        let mut num: usize = 0;

        for d in 0..md_end {
            num = bin[d];
            bin[d] = start;
            start += num;
        }

        for v in 1..n {
            pos[v] = bin[self.kcore[v]];
            self.kcore_order[pos[v]] = v;
            bin[self.kcore[v]] += 1;
        }

        for d in (1..md).rev() {
            bin[d] = bin[d - 1];
        }
        bin[0] = 1;

        let mut v: usize = 0;
        let mut u: usize = 0;

        // kcores
        for i in 1..n {
            v = self.kcore_order[i];
            for j in nodes[v - 1]..nodes[v] {
                u = edges[j] + 1;
                if (self.kcore[u] > self.kcore[v]) {
                    du = kcore[u];
                    pu = pos[u];
                    pw = bin[du];
                    w = kcore_order[pw];
                    if (u != w) {
                        pos[u] = pw;
                        kcore_order[pu] = w;
                        pos[w] = pu;
                        kcore_order[pw] = u;
                    }
                    //bin[du]++;   kcore[u]--;
                }
            }
        }
    }
}
