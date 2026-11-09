//  Copyright (c) 2026 Anshuman Agrawal
//  Copyright (c) 2026 Alexander Strack
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/collectives/detail/centered_tree.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using hpx::collectives::detail::centered_comm;
using hpx::collectives::detail::centered_leaf_size;
using hpx::collectives::detail::checked_data_size_product;
using hpx::collectives::detail::checked_data_size_sum;
using hpx::collectives::detail::is_ragged_rows_v;
using hpx::collectives::detail::is_uniform_rows_v;
using hpx::collectives::detail::make_centered_tree;
using hpx::collectives::detail::ragged_rows;
using hpx::collectives::detail::uniform_rows;

struct move_only_value
{
    explicit move_only_value(int const value)
      : value(value)
    {
    }

    move_only_value(move_only_value const&) = delete;
    move_only_value& operator=(move_only_value const&) = delete;

    move_only_value(move_only_value&& other) noexcept
      : value(other.value)
    {
        other.value = -1;
    }

    move_only_value& operator=(move_only_value&&) = delete;

    int value;
};

///////////////////////////////////////////////////////////////////////////////
// Topology of the centered k-ary communicator tree

// The sites covered by the subtree below `site`: the site itself plus the
// subtrees of the children of the communicator it roots.
std::size_t subtree_size(
    std::vector<centered_comm> const& tree, std::size_t const site)
{
    for (auto const& comm : tree)
    {
        if (comm.root == site)
        {
            std::size_t size = 0;
            for (std::size_t i = 0; i != comm.members.size(); ++i)
            {
                size += i == comm.root_rank ?
                    1 :
                    subtree_size(tree, comm.members[i]);
            }
            return size;
        }
    }
    return 1;
}

// Expand the subtree of `site` into its sites, in the order a hierarchical
// gather joins them (local rank order at every level).
void expand(std::vector<centered_comm> const& tree, std::size_t const site,
    std::vector<std::size_t>& out)
{
    for (auto const& comm : tree)
    {
        if (comm.root == site)
        {
            for (std::size_t i = 0; i != comm.members.size(); ++i)
            {
                if (i == comm.root_rank)
                {
                    out.push_back(site);
                }
                else
                {
                    expand(tree, comm.members[i], out);
                }
            }
            return;
        }
    }
    out.push_back(site);
}

void check_tree(
    std::size_t const n, std::size_t const arity, std::size_t const root = 0)
{
    auto const tree = make_centered_tree(n, arity, 4, root);
    std::size_t const leaf = centered_leaf_size(arity, 4);

    HPX_TEST(!tree.empty());
    if (tree.empty())
    {
        return;
    }

    std::vector<std::size_t> as_child(n, 0);
    std::vector<std::size_t> as_root(n, 0);

    for (auto const& comm : tree)
    {
        HPX_TEST(!comm.members.empty());
        HPX_TEST_EQ(comm.members.size(), comm.block_sizes.size());
        HPX_TEST_LT(comm.root_rank, comm.members.size());
        HPX_TEST_EQ(comm.members[comm.root_rank], comm.root);
        HPX_TEST_EQ(comm.block_sizes[comm.root_rank], std::size_t(1));
        HPX_TEST(std::is_sorted(comm.members.begin(), comm.members.end()));
        HPX_TEST(std::adjacent_find(comm.members.begin(), comm.members.end()) ==
            comm.members.end());

        bool const is_leaf = std::all_of(comm.block_sizes.begin(),
            comm.block_sizes.end(), [](std::size_t b) { return b == 1; });
        HPX_TEST_LTE(comm.members.size(), is_leaf ? leaf : arity + 1);

        // Every member represents its subtree, and the blocks of the members
        // are consecutive: member i+1's block starts right after member i's.
        for (std::size_t i = 0; i != comm.members.size(); ++i)
        {
            if (i != comm.root_rank)
            {
                HPX_TEST_EQ(
                    comm.block_sizes[i], subtree_size(tree, comm.members[i]));
                ++as_child[comm.members[i]];
            }
        }
        ++as_root[comm.root];
    }

    // the root site roots the first communicator and is nobody's child;
    // every other site is a non-root member of exactly one communicator
    HPX_TEST_EQ(tree.front().root, root);
    HPX_TEST_EQ(tree.front().depth, std::size_t(0));
    HPX_TEST_EQ(as_child[root], std::size_t(0));
    for (std::size_t s = 0; s != n; ++s)
    {
        if (s != root)
        {
            HPX_TEST_EQ(as_child[s], std::size_t(1));
        }
        HPX_TEST_LTE(as_root[s], std::size_t(1));
    }

    // The top-level blocks cover all sites.
    std::size_t total = 0;
    for (std::size_t const b : tree.front().block_sizes)
    {
        total += b;
    }
    HPX_TEST_EQ(total, n);

    // Joining the blocks in local rank order yields global site order (the
    // tree is an in-order search tree).
    std::vector<std::size_t> order;
    expand(tree, root, order);
    HPX_TEST_EQ(order.size(), n);
    for (std::size_t s = 0; s != order.size(); ++s)
    {
        HPX_TEST_EQ(order[s], s);
    }
}

void test_centered_tree_properties()
{
    for (std::size_t const arity : {2, 3, 4, 8, 32})
    {
        for (std::size_t const n : {1, 2, 3, 5, 16, 17, 255, 256})
        {
            check_tree(n, arity);
            for (std::size_t const root :
                {std::size_t(1), n / 3, n / 2, n - 2, n - 1})
            {
                if (root < n)
                {
                    check_tree(n, arity, root);
                }
            }
        }
    }

    // every root of smaller trees
    for (std::size_t n = 1; n <= 40; ++n)
    {
        for (std::size_t arity = 2; arity <= 9; ++arity)
        {
            for (std::size_t root = 0; root != n; ++root)
            {
                check_tree(n, arity, root);
            }
        }
    }
}

void test_centered_tree_example()
{
    // 16 sites, arity 2, leaf size 4 (root in brackets):
    //   ([0] 4 11)
    //   (2 [4] 6) (9 [11] 13)
    //   (1 [2] 3) (5 [6] 7) (8 [9] 10) (12 [13] 14 15)
    std::vector<std::vector<std::size_t>> const expected_members = {{0, 4, 11},
        {2, 4, 6}, {1, 2, 3}, {5, 6, 7}, {9, 11, 13}, {8, 9, 10},
        {12, 13, 14, 15}};
    std::vector<std::size_t> const expected_roots = {0, 4, 2, 6, 11, 9, 13};
    std::vector<std::vector<std::size_t>> const expected_blocks = {{1, 7, 8},
        {3, 1, 3}, {1, 1, 1}, {1, 1, 1}, {3, 1, 4}, {1, 1, 1}, {1, 1, 1, 1}};

    auto const tree = make_centered_tree(16, 2, 4);
    HPX_TEST_EQ(tree.size(), expected_members.size());
    for (std::size_t c = 0; c != tree.size() && c != expected_members.size();
        ++c)
    {
        HPX_TEST(tree[c].members == expected_members[c]);
        HPX_TEST_EQ(tree[c].root, expected_roots[c]);
        HPX_TEST(tree[c].block_sizes == expected_blocks[c]);
    }

    // The same sites with root 5: the 5 sites left of the root and the 10
    // sites right of it get one block each, no block spans the root.
    //   (2 [5] 10)
    //   (0 [2] 3) (7 [10] 13)
    //   ([0] 1) ([3] 4) (6 [7] 8 9) (11 [13] 14)
    //   ([11] 12) ([14] 15)
    std::vector<std::vector<std::size_t>> const expected_members_5 = {
        {2, 5, 10}, {0, 2, 3}, {0, 1}, {3, 4}, {7, 10, 13}, {6, 7, 8, 9},
        {11, 13, 14}, {11, 12}, {14, 15}};
    std::vector<std::size_t> const expected_roots_5 = {
        5, 2, 0, 3, 10, 7, 13, 11, 14};
    std::vector<std::size_t> const expected_root_ranks_5 = {
        1, 1, 0, 0, 1, 1, 1, 0, 0};
    std::vector<std::vector<std::size_t>> const expected_blocks_5 = {{5, 1, 10},
        {2, 1, 2}, {1, 1}, {1, 1}, {4, 1, 5}, {1, 1, 1, 1}, {2, 1, 2}, {1, 1},
        {1, 1}};

    auto const tree_5 = make_centered_tree(16, 2, 4, 5);
    HPX_TEST_EQ(tree_5.size(), expected_members_5.size());
    for (std::size_t c = 0;
        c != tree_5.size() && c != expected_members_5.size(); ++c)
    {
        HPX_TEST(tree_5[c].members == expected_members_5[c]);
        HPX_TEST_EQ(tree_5[c].root, expected_roots_5[c]);
        HPX_TEST_EQ(tree_5[c].root_rank, expected_root_ranks_5[c]);
        HPX_TEST(tree_5[c].block_sizes == expected_blocks_5[c]);
    }

    // with the last site as the root, all blocks are left of it
    auto const tree_15 = make_centered_tree(16, 2, 4, 15);
    HPX_TEST(tree_15.front().members == (std::vector<std::size_t>{3, 10, 15}));
    HPX_TEST_EQ(tree_15.front().root_rank, std::size_t(2));
    HPX_TEST(
        tree_15.front().block_sizes == (std::vector<std::size_t>{7, 8, 1}));

    // small trees are a single communicator rooted at the root site
    for (std::size_t n = 1; n <= 4; ++n)
    {
        for (std::size_t root = 0; root != n; ++root)
        {
            auto const flat = make_centered_tree(n, 2, 4, root);
            HPX_TEST_EQ(flat.size(), std::size_t(1));
            HPX_TEST_EQ(flat.front().members.size(), n);
            HPX_TEST_EQ(flat.front().root, root);
            HPX_TEST_EQ(flat.front().root_rank, root);
        }
    }

    // the leaf size grows with the arity
    HPX_TEST_EQ(make_centered_tree(9, 8).size(), std::size_t(1));
    HPX_TEST_LT(std::size_t(1), make_centered_tree(10, 8).size());
}

// Create the hierarchical communicator with the given root on all sites and
// check that every site holds the sub-communicators of the tree it belongs
// to: the one in which it is a child (if any), then the one it roots (if any).
void check_communicator_roots(
    std::size_t const n, std::size_t const arity, std::size_t const root)
{
    // hpx_main runs on every locality, each simulating all sites, so the
    // names must differ between the localities.
    std::string const basename = "/test/hierarchical_helpers/roots/" +
        std::to_string(hpx::get_locality_id()) + "/" + std::to_string(n) + "/" +
        std::to_string(arity) + "/";
    auto const tree = make_centered_tree(n, arity, 4, root);

    std::vector<hpx::future<void>> sites;
    for (std::size_t site = 0; site != n; ++site)
    {
        sites.push_back(hpx::async([&, site]() {
            auto const comms =
                hpx::collectives::create_hierarchical_communicator(
                    basename.c_str(), hpx::collectives::num_sites_arg(n),
                    hpx::collectives::this_site_arg(site),
                    hpx::collectives::arity_arg(arity),
                    hpx::collectives::generation_arg(),
                    hpx::collectives::root_site_arg(root),
                    hpx::collectives::flat_fallback_threshold_arg(0));

            HPX_TEST_EQ(comms.get_root_site(), root);
            HPX_TEST_EQ(comms.is_flat(), tree.size() == 1);
            HPX_TEST(comms.top_level_block_sizes() == tree.front().block_sizes);

            std::vector<std::size_t> local_ranks;
            std::ptrdiff_t top_level_group = -1;
            for (std::size_t c = 0; c != tree.size(); ++c)
            {
                auto const& members = tree[c].members;
                auto const it = std::find(members.begin(), members.end(), site);
                if (it != members.end())
                {
                    std::size_t const local = static_cast<std::size_t>(
                        std::distance(members.begin(), it));
                    if (c == 0)
                    {
                        top_level_group = static_cast<std::ptrdiff_t>(local);
                    }
                    if (local_ranks.size() < comms.size())
                    {
                        HPX_TEST(comms.block_sizes(local_ranks.size()) ==
                            tree[c].block_sizes);
                    }
                    local_ranks.push_back(local);
                }
            }

            HPX_TEST_EQ(comms.top_level_group(), top_level_group);
            HPX_TEST_EQ(comms.size(), local_ranks.size());
            for (std::size_t i = 0;
                i != local_ranks.size() && i != comms.size(); ++i)
            {
                HPX_TEST_EQ(comms.site(i), local_ranks[i]);
            }

            // The root only belongs to the top-level communicator, where it is
            // the root of the communicator.
            if (site == root)
            {
                HPX_TEST_EQ(comms.size(), std::size_t(1));
                HPX_TEST_EQ(comms.site(0), tree.front().root_rank);
                HPX_TEST(comms.get(0).is_root());
            }
        }));
    }
    hpx::wait_all(sites);
    for (auto& f : sites)
    {
        HPX_TEST(!f.has_exception());
    }
}

void test_communicator_roots()
{
    for (std::size_t n = 1; n <= 12; ++n)
    {
        for (std::size_t arity = 2; arity <= 4; ++arity)
        {
            for (std::size_t root = 0; root != n; ++root)
            {
                check_communicator_roots(n, arity, root);
            }
        }
    }
    for (std::size_t const root : {0, 1, 8, 15, 16})
    {
        check_communicator_roots(17, 2, root);
    }
}

void test_uniform_rows_explicit_slices()
{
    uniform_rows<int> rows(std::vector<int>{0, 1, 2, 3, 4, 5, 6}, 7);
    rows.set_slices({3, 1, 3});
    HPX_TEST(rows.is_valid());

    std::vector<int> const values{0, 1, 2, 3, 4, 5, 6};
    std::vector<std::size_t> const slices{3, 1, 3};
    std::size_t first = 0;
    for (std::size_t slice = 0; slice != slices.size(); ++slice)
    {
        uniform_rows<int> copy(std::vector<int>(values), 7);
        copy.set_slices(slices);
        auto part = HPX_MOVE(copy).extract(slice, slices.size());
        HPX_TEST_EQ(part.num_rows(), slices[slice]);
        for (std::size_t i = 0; i != slices[slice]; ++i)
        {
            HPX_TEST_EQ(part.data()[i], values[first + i]);
        }
        first += slices[slice];
    }

    // the slice table survives serialization
    std::vector<char> buffer;
    hpx::serialization::output_archive output(buffer);
    output << rows;

    uniform_rows<int> restored;
    hpx::serialization::input_archive input(buffer);
    input >> restored;
    auto last = HPX_MOVE(restored).extract(2, 3);
    HPX_TEST_EQ(last.num_rows(), std::size_t(3));
    HPX_TEST_EQ(last.data()[0], 4);
}

void test_is_uniform_rows_trait()
{
    static_assert(is_uniform_rows_v<uniform_rows<int>>);
    static_assert(is_uniform_rows_v<uniform_rows<int> const>);
    static_assert(is_uniform_rows_v<uniform_rows<int>&>);
    static_assert(is_uniform_rows_v<uniform_rows<int> const&>);
    static_assert(!is_uniform_rows_v<int>);
    static_assert(!is_uniform_rows_v<std::vector<int>>);
}

void test_is_ragged_rows_trait()
{
    static_assert(is_ragged_rows_v<ragged_rows<int>>);
    static_assert(is_ragged_rows_v<ragged_rows<int> const>);
    static_assert(is_ragged_rows_v<ragged_rows<int>&>);
    static_assert(is_ragged_rows_v<ragged_rows<int> const&>);
    static_assert(!is_ragged_rows_v<int>);
    static_assert(!is_ragged_rows_v<std::vector<int>>);
    static_assert(std::is_nothrow_constructible_v<ragged_rows<int>,
        std::vector<int>&&, std::vector<std::size_t>&&>);
}

void test_ragged_rows_extract_columns()
{
    std::vector<ragged_rows<int>> values;
    values.emplace_back(std::vector<int>{3, 4, 103, 104, 203, 204},
        std::vector<std::size_t>{0, 0, 2, 2, 4, 4, 6});
    values.emplace_back(std::vector<int>{300, 301, 302, 400, 401, 402},
        std::vector<std::size_t>{0, 3, 3, 6, 6});

    ragged_rows<int> column_zero(values, 0);
    HPX_TEST_EQ(column_zero.data().size(), static_cast<std::size_t>(6));
    HPX_TEST_EQ(column_zero.offsets().size(), static_cast<std::size_t>(3));
    HPX_TEST_EQ(column_zero.offsets()[0], static_cast<std::size_t>(0));
    HPX_TEST_EQ(column_zero.offsets()[1], static_cast<std::size_t>(0));
    HPX_TEST_EQ(column_zero.offsets()[2], static_cast<std::size_t>(6));
    for (std::size_t i = 0; i != column_zero.data().size(); ++i)
    {
        HPX_TEST_EQ(column_zero.data()[i],
            static_cast<int>(300 + (i / 3) * 100 + i % 3));
    }

    ragged_rows<int> column_one(values, 1);
    HPX_TEST_EQ(column_one.data().size(), static_cast<std::size_t>(6));
    HPX_TEST_EQ(column_one.offsets().size(), static_cast<std::size_t>(3));
    HPX_TEST_EQ(column_one.offsets()[0], static_cast<std::size_t>(0));
    HPX_TEST_EQ(column_one.offsets()[1], static_cast<std::size_t>(6));
    HPX_TEST_EQ(column_one.offsets()[2], static_cast<std::size_t>(6));
    for (std::size_t i = 0; i != column_one.data().size(); ++i)
    {
        HPX_TEST_EQ(
            column_one.data()[i], static_cast<int>((i / 2) * 100 + 3 + i % 2));
    }

    std::vector<ragged_rows<move_only_value>> move_only_values;
    std::vector<move_only_value> first;
    first.emplace_back(1);
    first.emplace_back(2);
    move_only_values.emplace_back(
        HPX_MOVE(first), std::vector<std::size_t>{0, 1, 2});
    std::vector<move_only_value> second;
    second.emplace_back(3);
    second.emplace_back(4);
    move_only_values.emplace_back(
        HPX_MOVE(second), std::vector<std::size_t>{0, 1, 2});

    ragged_rows<move_only_value> move_only_column(move_only_values, 1);
    HPX_TEST_EQ(move_only_column.data().size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(move_only_column.data()[0].value, 2);
    HPX_TEST_EQ(move_only_column.data()[1].value, 4);
    auto move_only_data = HPX_MOVE(move_only_column).release_data();
    HPX_TEST_EQ(move_only_data.size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(move_only_data[1].value, 4);
}

void test_ragged_rows_validation()
{
    ragged_rows<int> default_rows;
    ragged_rows<int> valid_empty_rows(
        std::vector<int>{}, std::vector<std::size_t>{0, 0, 0});
    HPX_TEST(default_rows.is_valid());
    HPX_TEST_EQ(default_rows.num_segments(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(default_rows.offsets().size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(default_rows.offsets()[0], static_cast<std::size_t>(0));
    HPX_TEST_EQ(default_rows.offsets()[1], static_cast<std::size_t>(0));
    HPX_TEST_NO_THROW(default_rows.validate("test_ragged_rows_validation"));
    HPX_TEST(valid_empty_rows.is_valid());
    HPX_TEST_NO_THROW(valid_empty_rows.validate("test_ragged_rows_validation"));
    HPX_TEST_NO_THROW(valid_empty_rows.validate_for_exchange(
        2, "test_ragged_rows_validation"));

    auto deserialize_rows = [](std::vector<int> data,
                                std::vector<std::size_t> offsets) {
        std::vector<char> buffer;
        hpx::serialization::output_archive output(buffer);
        output << data << offsets;

        ragged_rows<int> rows;
        hpx::serialization::input_archive input(buffer);
        input >> rows;
        return rows;
    };

    auto empty_offsets = deserialize_rows({}, {});
    auto single_offset = deserialize_rows({}, {0});
    auto nonzero_first = deserialize_rows({1}, {1, 1});
    auto decreasing = deserialize_rows({1, 2}, {0, 2, 1, 2});
    auto wrong_end = deserialize_rows({1, 2}, {0, 1});
    ragged_rows<int> incomplete_exchange(
        std::vector<int>{1}, std::vector<std::size_t>{0, 1});

    HPX_TEST(!empty_offsets.is_valid());
    HPX_TEST(!single_offset.is_valid());
    HPX_TEST(!nonzero_first.is_valid());
    HPX_TEST(!decreasing.is_valid());
    HPX_TEST(!wrong_end.is_valid());
    HPX_TEST_THROW(
        empty_offsets.validate("test_ragged_rows_validation"), hpx::exception);
    HPX_TEST_THROW(
        single_offset.validate("test_ragged_rows_validation"), hpx::exception);
    HPX_TEST_THROW(
        nonzero_first.validate("test_ragged_rows_validation"), hpx::exception);
    HPX_TEST_THROW(
        decreasing.validate("test_ragged_rows_validation"), hpx::exception);
    HPX_TEST_THROW(
        wrong_end.validate("test_ragged_rows_validation"), hpx::exception);
    HPX_TEST_THROW(incomplete_exchange.validate_for_exchange(
                       2, "test_ragged_rows_validation"),
        hpx::exception);
}

void test_uniform_rows_slicing()
{
    uniform_rows<int> first(std::vector<int>{10, 11, 20, 21, 30, 31}, 3);
    auto left = HPX_MOVE(first).extract(0, 2);
    HPX_TEST_EQ(left.data().size(), static_cast<std::size_t>(4));
    HPX_TEST_EQ(left.num_rows(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(left.data()[0], 10);
    HPX_TEST_EQ(left.data()[1], 11);
    HPX_TEST_EQ(left.data()[2], 20);
    HPX_TEST_EQ(left.data()[3], 21);

    uniform_rows<int> second(std::vector<int>{10, 11, 20, 21, 30, 31}, 3);
    auto right = HPX_MOVE(second).extract(1, 2);
    HPX_TEST_EQ(right.data().size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(right.num_rows(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(right.data()[0], 30);
    HPX_TEST_EQ(right.data()[1], 31);

    uniform_rows<int> empty_rows(std::vector<int>{}, 3);
    auto empty = HPX_MOVE(empty_rows).extract(1, 3);
    HPX_TEST(empty.data().empty());
    HPX_TEST_EQ(empty.num_rows(), static_cast<std::size_t>(1));

    std::vector<move_only_value> move_only_values;
    move_only_values.reserve(3);
    move_only_values.emplace_back(10);
    move_only_values.emplace_back(20);
    move_only_values.emplace_back(30);
    uniform_rows<move_only_value> move_only_rows(HPX_MOVE(move_only_values), 3);
    auto middle = HPX_MOVE(move_only_rows).extract(1, 3);
    HPX_TEST_EQ(middle.data().size(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(middle.data()[0].value, 20);
}

void test_uniform_rows_extract_single_slice()
{
    uniform_rows<int> rows(std::vector<int>{1, 2, 3, 4}, 2);
    auto slice = HPX_MOVE(rows).extract(0, 1);
    HPX_TEST_EQ(slice.num_rows(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(slice.data().size(), static_cast<std::size_t>(4));
    HPX_TEST_EQ(slice.data()[0], 1);
    HPX_TEST_EQ(slice.data()[1], 2);
    HPX_TEST_EQ(slice.data()[2], 3);
    HPX_TEST_EQ(slice.data()[3], 4);
}

void test_uniform_rows_merge()
{
    std::vector<uniform_rows<std::string>> values;
    values.emplace_back(std::vector<std::string>{"a", "b", "c", "d"}, 2);
    values.emplace_back(std::vector<std::string>{"e", "f"}, 1);

    uniform_rows<std::string> result(HPX_MOVE(values));
    HPX_TEST_EQ(result.data().size(), static_cast<std::size_t>(6));
    HPX_TEST_EQ(result.num_rows(), static_cast<std::size_t>(3));
    for (std::size_t i = 0; i != result.data().size(); ++i)
    {
        HPX_TEST_EQ(
            result.data()[i], std::string(1, static_cast<char>('a' + i)));
    }

    std::vector<uniform_rows<int>> mismatched;
    mismatched.emplace_back(std::vector<int>{1, 2}, 1);
    mismatched.emplace_back(std::vector<int>{3, 4, 5}, 1);
    HPX_TEST_THROW(
        (void) uniform_rows<int>(HPX_MOVE(mismatched)), hpx::exception);

    std::vector<uniform_rows<int>> singleton;
    singleton.emplace_back(std::vector<int>{1, 2, 3, 4}, 2);
    int const* const original_data = singleton.front().data().data();
    uniform_rows<int> singleton_result(HPX_MOVE(singleton));
    HPX_TEST_EQ(singleton_result.data().data(), original_data);
    HPX_TEST_EQ(singleton_result.num_rows(), static_cast<std::size_t>(2));
}

void test_uniform_rows_merge_move_only()
{
    std::vector<uniform_rows<move_only_value>> values;

    std::vector<move_only_value> first_row;
    first_row.emplace_back(1);
    first_row.emplace_back(2);
    values.emplace_back(HPX_MOVE(first_row), 1);

    std::vector<move_only_value> second_row;
    second_row.emplace_back(3);
    second_row.emplace_back(4);
    values.emplace_back(HPX_MOVE(second_row), 1);

    uniform_rows<move_only_value> merged(HPX_MOVE(values));
    HPX_TEST_EQ(merged.data().size(), static_cast<std::size_t>(4));
    HPX_TEST_EQ(merged.num_rows(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(merged.data()[0].value, 1);
    HPX_TEST_EQ(merged.data()[1].value, 2);
    HPX_TEST_EQ(merged.data()[2].value, 3);
    HPX_TEST_EQ(merged.data()[3].value, 4);
}

void test_uniform_rows_empty_merge()
{
    std::vector<uniform_rows<int>> empty_values;
    uniform_rows<int> result(HPX_MOVE(empty_values));
    HPX_TEST_EQ(result.num_rows(), static_cast<std::size_t>(0));
    HPX_TEST(result.data().empty());
}

void test_uniform_rows_vector_bool()
{
    std::vector<uniform_rows<bool>> values;
    values.emplace_back(std::vector<bool>{true, false, true, true}, 2);
    values.emplace_back(std::vector<bool>{false, false}, 1);

    uniform_rows<bool> merged(HPX_MOVE(values));
    HPX_TEST_EQ(merged.data().size(), static_cast<std::size_t>(6));
    HPX_TEST_EQ(merged.num_rows(), static_cast<std::size_t>(3));
    HPX_TEST(merged.data()[0]);
    HPX_TEST(!merged.data()[1]);
    HPX_TEST(merged.data()[2]);
    HPX_TEST(merged.data()[3]);
    HPX_TEST(!merged.data()[4]);
    HPX_TEST(!merged.data()[5]);

    auto slice = HPX_MOVE(merged).extract(1, 3);
    HPX_TEST_EQ(slice.data().size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(slice.num_rows(), static_cast<std::size_t>(1));
    HPX_TEST(slice.data()[0]);
    HPX_TEST(slice.data()[1]);

    uniform_rows<bool> value(true);
    HPX_TEST(HPX_MOVE(value).unwrap_value());
}

void test_uniform_rows_validation()
{
    uniform_rows<int> valid_empty_rows(std::vector<int>{}, 3);
    uniform_rows<int> valid_no_rows(std::vector<int>{}, 0);
    uniform_rows<int> valid_rows(std::vector<int>{1, 2, 3, 4}, 2);

    auto deserialize_rows = [](std::vector<int> data, std::size_t num_rows) {
        std::vector<char> buffer;
        hpx::serialization::output_archive output(buffer);
        output << data << num_rows << std::vector<std::size_t>{};

        uniform_rows<int> rows;
        hpx::serialization::input_archive input(buffer);
        input >> rows;
        return rows;
    };

    auto rows_with_no_width = deserialize_rows(std::vector<int>{1}, 0);
    auto nonuniform_width = deserialize_rows(std::vector<int>{1, 2, 3}, 2);

    HPX_TEST(valid_empty_rows.is_valid());
    HPX_TEST(valid_no_rows.is_valid());
    HPX_TEST(valid_rows.is_valid());
    HPX_TEST_EQ(valid_empty_rows.row_width(), static_cast<std::size_t>(0));
    HPX_TEST_EQ(valid_rows.row_width(), static_cast<std::size_t>(2));
    HPX_TEST(!rows_with_no_width.is_valid());
    HPX_TEST(!nonuniform_width.is_valid());
    HPX_TEST_NO_THROW(
        valid_empty_rows.validate("test_uniform_rows_validation"));
    HPX_TEST_THROW(rows_with_no_width.validate("test_uniform_rows_validation"),
        hpx::exception);
    HPX_TEST_THROW(nonuniform_width.validate("test_uniform_rows_validation"),
        hpx::exception);
}

void test_uniform_rows_row_width_zero_rows()
{
    uniform_rows<int> zero_rows(std::vector<int>{}, 0);
    HPX_TEST_EQ(zero_rows.row_width(), static_cast<std::size_t>(0));
}

void test_uniform_rows_construction()
{
    using extract_type = decltype(&uniform_rows<int>::extract);
    static_assert(std::is_invocable_v<extract_type, uniform_rows<int>&&,
        std::size_t, std::size_t>);
    static_assert(!std::is_invocable_v<extract_type, uniform_rows<int>&,
        std::size_t, std::size_t>);
    static_assert(std::is_nothrow_constructible_v<uniform_rows<int>,
        std::vector<int>&&, std::size_t>);
    static_assert(
        std::is_nothrow_constructible_v<uniform_rows<int>, std::vector<int>&&>);

    uniform_rows<int> rows(std::vector<int>{1, 2, 3});
    HPX_TEST_EQ(rows.data().size(), static_cast<std::size_t>(3));
    HPX_TEST_EQ(rows.num_rows(), static_cast<std::size_t>(3));
    HPX_TEST_EQ(rows.row_width(), static_cast<std::size_t>(1));

    std::string value = "value";
    uniform_rows<std::string> copied_value(value);
    HPX_TEST_EQ(HPX_MOVE(copied_value).unwrap_value(), value);

    uniform_rows<std::string> moved_value(std::string("moved"));
    HPX_TEST_EQ(HPX_MOVE(moved_value).unwrap_value(), std::string("moved"));

    uniform_rows<std::string> row(
        std::vector<std::string>{"first", "second"}, 1);
    auto unwrapped = HPX_MOVE(row).unwrap_row();
    HPX_TEST_EQ(unwrapped.size(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(unwrapped[0], std::string("first"));
    HPX_TEST_EQ(unwrapped[1], std::string("second"));

    uniform_rows<int> scalar_rows(std::vector<int>{1, 2, 3});
    auto values = HPX_MOVE(scalar_rows).unwrap_values();
    HPX_TEST_EQ(values.size(), static_cast<std::size_t>(3));
    HPX_TEST_EQ(values[0], 1);
    HPX_TEST_EQ(values[1], 2);
    HPX_TEST_EQ(values[2], 3);

    uniform_rows<int> releasable(std::vector<int>{4, 5, 6, 7}, 2);
    auto released = HPX_MOVE(releasable).release_data();
    HPX_TEST_EQ(released.size(), static_cast<std::size_t>(4));
}

void test_data_size_overflow()
{
    constexpr std::size_t max_size = (std::numeric_limits<std::size_t>::max)();

    HPX_TEST_EQ(checked_data_size_sum(max_size - 1, 1), max_size);
    HPX_TEST_EQ(checked_data_size_product(max_size, 1), max_size);
    HPX_TEST_EQ(checked_data_size_product(max_size, 0), std::size_t{0});
    HPX_TEST_THROW((void) checked_data_size_sum(max_size, 1), hpx::exception);
    HPX_TEST_THROW(
        (void) checked_data_size_product(max_size, 2), hpx::exception);
}

void test_uniform_rows_serialization()
{
    uniform_rows<std::string> rows(
        std::vector<std::string>{"zero", "one", "two", "three"}, 2);
    std::vector<char> rows_buffer;
    hpx::serialization::output_archive rows_output(rows_buffer);
    rows_output << rows;

    uniform_rows<std::string> restored_rows;
    hpx::serialization::input_archive rows_input(rows_buffer);
    rows_input >> restored_rows;
    HPX_TEST_EQ(restored_rows.data().size(), rows.data().size());
    for (std::size_t i = 0; i != rows.data().size(); ++i)
    {
        HPX_TEST_EQ(restored_rows.data()[i], rows.data()[i]);
    }
    HPX_TEST_EQ(restored_rows.num_rows(), rows.num_rows());
}

void test_ragged_rows_serialization()
{
    ragged_rows<std::string> empty_rows;
    std::vector<char> empty_buffer;
    hpx::serialization::output_archive empty_output(empty_buffer);
    empty_output << empty_rows;

    ragged_rows<std::string> restored_empty_rows;
    hpx::serialization::input_archive empty_input(empty_buffer);
    empty_input >> restored_empty_rows;
    HPX_TEST(restored_empty_rows.is_valid());
    HPX_TEST_EQ(
        restored_empty_rows.num_segments(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(
        restored_empty_rows.offsets().size(), static_cast<std::size_t>(2));

    ragged_rows<std::string> rows(
        std::vector<std::string>{"zero", "one", "two"},
        std::vector<std::size_t>{0, 2, 3});
    std::vector<char> buffer;
    hpx::serialization::output_archive output(buffer);
    output << rows;

    ragged_rows<std::string> restored;
    hpx::serialization::input_archive input(buffer);
    input >> restored;
    HPX_TEST_EQ(restored.data().size(), rows.data().size());
    HPX_TEST_EQ(restored.offsets().size(), rows.offsets().size());
    for (std::size_t i = 0; i != rows.data().size(); ++i)
    {
        HPX_TEST_EQ(restored.data()[i], rows.data()[i]);
    }
    for (std::size_t i = 0; i != rows.offsets().size(); ++i)
    {
        HPX_TEST_EQ(restored.offsets()[i], rows.offsets()[i]);
    }
}

int hpx_main()
{
    test_centered_tree_properties();
    test_centered_tree_example();
    test_communicator_roots();
    test_uniform_rows_explicit_slices();
    test_is_ragged_rows_trait();
    test_is_uniform_rows_trait();
    test_ragged_rows_extract_columns();
    test_ragged_rows_validation();
    test_uniform_rows_slicing();
    test_uniform_rows_extract_single_slice();
    test_uniform_rows_merge();
    test_uniform_rows_merge_move_only();
    test_uniform_rows_empty_merge();
    test_uniform_rows_vector_bool();
    test_uniform_rows_validation();
    test_uniform_rows_row_width_zero_rows();
    test_uniform_rows_construction();
    test_data_size_overflow();
    test_ragged_rows_serialization();
    test_uniform_rows_serialization();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

#endif
