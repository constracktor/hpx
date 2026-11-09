//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file detail/centered_tree.hpp
///
/// Topology of the hierarchical communicator tree ("centered" k-ary tree). It
/// depends only on the standard library (and the HPX configuration) so it can
/// be tested on its own.
///
/// The root site r is the root of the tree. The other sites are split into k
/// blocks of consecutive sites whose roots (centers) are the children of r:
/// the sites left of r and the sites right of r are split separately (no
/// block spans r), with the k blocks shared between both sides so that the
/// largest block is as small as possible. For r = 0 the sites 1..n-1 are
/// simply split evenly into k blocks. A block with more than
/// L = max(leaf_size, k + 1) sites is rooted at its center c; the floor(k/2)
/// children of c are the centers of an even split of the sites left of c, the
/// ceil(k/2) other children the centers of an even split of the sites right
/// of c. A block of at most L sites becomes one leaf communicator holding the
/// whole block. Every communicator thus spans a parent and its (at most k)
/// children, or one leaf block. If all n sites fit into one leaf, the tree is
/// a single (flat) communicator rooted at r.
///
/// The tree is an in-order search tree: every subtree covers a block of
/// consecutive sites, and its root lies between the blocks of its left and
/// right children. Each communicator lists its members in ascending site
/// order, with the root at local rank `root_rank`. Joining per-member
/// contributions in local rank order therefore yields global site order for
/// any root, which is what gather, scatter, the scans, all_to_all and
/// non-commutative reductions rely on.
///
/// Every site is a non-root member of at most one communicator (the root
/// site of none) and roots at most one communicator.
///
/// For 16 sites, k = 2 and leaf_size 4 this produces (root in brackets)
///   root 0, level 0: ([0] 4 11)
///           level 1: (2 [4] 6) (9 [11] 13)
///           level 2: (1 [2] 3) (5 [6] 7) (8 [9] 10) (12 [13] 14 15)
///   root 5, level 0: (2 [5] 10)
///           level 1: (0 [2] 3) (7 [10] 13)
///           level 2: ([0] 1) ([3] 4) (6 [7] 8 9) (11 [13] 14)
///           level 3: ([11] 12) ([14] 15)

#pragma once

#include <hpx/config.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace hpx::collectives::detail {

    HPX_CXX_EXPORT struct centered_comm
    {
        std::size_t depth = 0;
        // the site rooting this communicator
        std::size_t root = 0;
        // local rank of the root, members[root_rank] == root
        std::size_t root_rank = 0;
        // all members in ascending site order
        std::vector<std::size_t> members;
        // number of sites in the subtree each member represents in this
        // communicator (1 for the root and for the members of a leaf)
        std::vector<std::size_t> block_sizes;
    };

    namespace centered_tree_impl {

        // The root of the block [left, right] (internal node or leaf). With
        // an even arity this is the plain center; with an odd arity the root
        // is shifted so the floor(k/2) left children and ceil(k/2) right
        // children get blocks of similar size.
        [[nodiscard]] constexpr std::size_t block_root(std::size_t const left,
            std::size_t const right, std::size_t const arity) noexcept
        {
            return left + ((arity / 2) * (right - left)) / arity;
        }

        // Split [left, right] into `count` blocks of consecutive sites whose
        // sizes differ by at most one, smaller blocks first. Empty blocks are
        // skipped.
        inline void split(std::vector<std::pair<std::size_t, std::size_t>>& out,
            std::size_t const left, std::size_t const right,
            std::size_t const count)
        {
            if (right < left || count == 0)
            {
                return;
            }

            std::size_t const size = right - left + 1;
            std::size_t const base = size / count;
            std::size_t const larger_from = count - size % count;

            std::size_t first = left;
            for (std::size_t i = 0; i != count; ++i)
            {
                std::size_t const block = base + (i >= larger_from ? 1 : 0);
                if (block != 0)
                {
                    out.emplace_back(first, first + block - 1);
                    first += block;
                }
            }
        }

        inline void fill(std::vector<centered_comm>& out, std::size_t left,
            std::size_t right, std::size_t depth, std::size_t arity,
            std::size_t leaf_size);

        // Add the communicator of `root` and its children, which are the
        // centers of `num_left_blocks` blocks splitting [left, root - 1] and
        // of `arity - num_left_blocks` blocks splitting [root + 1, right],
        // then the subtrees of the children.
        inline void fill_parent(std::vector<centered_comm>& out,
            std::size_t const left, std::size_t const right,
            std::size_t const root, std::size_t const num_left_blocks,
            std::size_t const depth, std::size_t const arity,
            std::size_t const leaf_size)
        {
            centered_comm comm;
            comm.depth = depth;
            comm.root = root;

            std::vector<std::pair<std::size_t, std::size_t>> blocks;
            if (root != left)
            {
                split(blocks, left, root - 1, num_left_blocks);
            }
            std::size_t const num_left = blocks.size();
            split(blocks, root + 1, right, arity - num_left_blocks);

            for (std::size_t i = 0; i != blocks.size(); ++i)
            {
                if (i == num_left)
                {
                    comm.root_rank = comm.members.size();
                    comm.members.push_back(root);
                    comm.block_sizes.push_back(1);
                }
                auto const [l, r] = blocks[i];
                comm.members.push_back(block_root(l, r, arity));
                comm.block_sizes.push_back(r - l + 1);
            }
            if (num_left == blocks.size())
            {
                comm.root_rank = comm.members.size();
                comm.members.push_back(root);
                comm.block_sizes.push_back(1);
            }
            out.push_back(std::move(comm));

            for (auto const& [l, r] : blocks)
            {
                fill(out, l, r, depth + 1, arity, leaf_size);
            }
        }

        inline void fill(std::vector<centered_comm>& out,
            std::size_t const left, std::size_t const right,
            std::size_t const depth, std::size_t const arity,
            std::size_t const leaf_size)
        {
            std::size_t const size = right - left + 1;
            if (size <= 1)
            {
                // a single site needs no communicator of its own
                return;
            }

            std::size_t const root = block_root(left, right, arity);
            if (size <= leaf_size)
            {
                centered_comm comm;
                comm.depth = depth;
                comm.root = root;
                for (std::size_t s = left; s <= right; ++s)
                {
                    comm.members.push_back(s);
                    comm.block_sizes.push_back(1);
                }
                comm.root_rank = root - left;
                out.push_back(std::move(comm));
                return;
            }

            fill_parent(
                out, left, right, root, arity / 2, depth, arity, leaf_size);
        }

        // The number of the k child blocks of the root site that split the
        // sites left of it (the others split the sites right of it): every
        // non-empty side gets at least one block, and the largest block is as
        // small as possible.
        [[nodiscard]] constexpr std::size_t top_level_left_blocks(
            std::size_t const num_left, std::size_t const num_right,
            std::size_t const arity) noexcept
        {
            if (num_left == 0)
            {
                return 0;
            }
            if (num_right == 0)
            {
                return arity;
            }

            std::size_t best = 1;
            std::size_t best_size = 0;
            for (std::size_t k_left = 1; k_left < arity; ++k_left)
            {
                std::size_t const k_right = arity - k_left;
                std::size_t const largest =
                    (std::max) ((num_left + k_left - 1) / k_left,
                        (num_right + k_right - 1) / k_right);
                if (best_size == 0 || largest < best_size)
                {
                    best = k_left;
                    best_size = largest;
                }
            }
            return best;
        }
    }    // namespace centered_tree_impl

    // The effective leaf size: a leaf must be able to hold at least a parent
    // and its k children.
    HPX_CXX_EXPORT [[nodiscard]] constexpr std::size_t centered_leaf_size(
        std::size_t const arity, std::size_t const leaf_size) noexcept
    {
        return (std::max) (leaf_size, arity + 1);
    }

    // All communicators of the tree rooted at site `root`, parents before
    // their children. An arity below 2 is treated as 2.
    HPX_CXX_EXPORT [[nodiscard]] inline std::vector<centered_comm>
    make_centered_tree(std::size_t const num_sites, std::size_t arity = 2,
        std::size_t leaf_size = 4, std::size_t const root = 0)
    {
        if (arity < 2)
        {
            arity = 2;
        }
        leaf_size = centered_leaf_size(arity, leaf_size);

        std::vector<centered_comm> out;
        if (num_sites == 0 || root >= num_sites)
        {
            return out;
        }

        if (num_sites <= leaf_size)
        {
            centered_comm comm;
            comm.root = root;
            comm.root_rank = root;
            for (std::size_t s = 0; s != num_sites; ++s)
            {
                comm.members.push_back(s);
                comm.block_sizes.push_back(1);
            }
            out.push_back(std::move(comm));
            return out;
        }

        // The root's k child blocks are shared between the sites left and
        // right of it (empty blocks are skipped).
        centered_tree_impl::fill_parent(out, 0, num_sites - 1, root,
            centered_tree_impl::top_level_left_blocks(
                root, num_sites - 1 - root, arity),
            0, arity, leaf_size);
        return out;
    }
}    // namespace hpx::collectives::detail
