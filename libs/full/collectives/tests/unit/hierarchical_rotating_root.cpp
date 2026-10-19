//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// A hierarchical communicator can be rooted at any site. This runs every
// hierarchical collective on one shared communicator instance for every choice
// of root and checks that the results are the same as for root 0: per-site
// vectors stay indexed by the actual site numbers, and the scans keep their
// site order.

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

void run_site(std::string const& basename, std::uint32_t const num_sites,
    std::uint32_t const site, std::uint32_t const root, std::size_t const arity,
    std::size_t const threshold)
{
    using u32 = std::uint32_t;
    std::string const ctx = "N=" + std::to_string(num_sites) +
        " arity=" + std::to_string(arity) +
        " thr=" + std::to_string(threshold) + " root=" + std::to_string(root);

    auto const comms = create_hierarchical_communicator(basename.c_str(),
        num_sites_arg(num_sites), this_site_arg(site), arity_arg(arity),
        generation_arg(), root_site_arg(root),
        flat_fallback_threshold_arg(threshold));

    HPX_TEST_EQ(static_cast<std::size_t>(comms.get_root_site()),
        static_cast<std::size_t>(root));

    std::size_t gen = 0;
    for (u32 i = 0; i != 3; ++i)
    {
        // broadcast
        {
            ++gen;
            u32 got = 0;
            if (site == root)
            {
                got = broadcast_to(comms, u32(1000 + i), this_site_arg(site),
                    generation_arg(gen))
                          .get();
            }
            else
            {
                got = broadcast_from<u32>(
                    comms, this_site_arg(site), generation_arg(gen))
                          .get();
            }
            HPX_TEST_EQ(got, u32(1000 + i));
        }

        // reduce
        {
            ++gen;
            if (site == root)
            {
                u32 const sum = reduce_here(comms, u32(site + i),
                    std::plus<u32>{}, this_site_arg(site), generation_arg(gen))
                                    .get();
                u32 expected = 0;
                for (u32 j = 0; j != num_sites; ++j)
                {
                    expected += j + i;
                }
                HPX_TEST_EQ(sum, expected);
            }
            else
            {
                reduce_there(comms, u32(site + i), std::plus<u32>{},
                    this_site_arg(site), generation_arg(gen))
                    .get();
            }
        }

        // gather: the result is in site order
        {
            ++gen;
            if (site == root)
            {
                std::vector<u32> const got = gather_here(comms,
                    u32(site * 7 + i), this_site_arg(site), generation_arg(gen))
                                                 .get();
                HPX_TEST_EQ(got.size(), std::size_t(num_sites));
                for (u32 j = 0; j != got.size(); ++j)
                {
                    HPX_TEST_EQ(got[j], u32(j * 7 + i));
                }
            }
            else
            {
                gather_there(comms, u32(site * 7 + i), this_site_arg(site),
                    generation_arg(gen))
                    .get();
            }
        }

        // scatter: values[j] goes to site j
        {
            ++gen;
            u32 got = 0;
            if (site == root)
            {
                std::vector<u32> values(num_sites);
                for (u32 j = 0; j != num_sites; ++j)
                {
                    values[j] = 500 + j * 3 + i;
                }
                got = scatter_to(comms, std::move(values), this_site_arg(site),
                    generation_arg(gen))
                          .get();
            }
            else
            {
                got = scatter_from<u32>(
                    comms, this_site_arg(site), generation_arg(gen))
                          .get();
            }
            HPX_TEST_EQ(got, u32(500 + site * 3 + i));
        }

        // all_gather
        {
            ++gen;
            std::vector<u32> const got = all_gather(hpx::launch::sync, comms,
                u32(site * 5 + i), this_site_arg(site), generation_arg(gen));
            HPX_TEST_EQ(got.size(), std::size_t(num_sites));
            for (u32 j = 0; j != got.size(); ++j)
            {
                HPX_TEST_EQ_MSG(got[j], u32(j * 5 + i),
                    ctx + " all_gather site=" + std::to_string(site));
            }
        }

        // all_reduce
        {
            ++gen;
            u32 const sum = all_reduce(hpx::launch::sync, comms, u32(site + i),
                std::plus<u32>{}, this_site_arg(site), generation_arg(gen));
            u32 expected = 0;
            for (u32 j = 0; j != num_sites; ++j)
            {
                expected += j + i;
            }
            HPX_TEST_EQ(sum, expected);
        }

        // all_to_all: result[s] is what site s sent to this site
        {
            ++gen;
            std::vector<u32> values(num_sites);
            for (u32 d = 0; d != num_sites; ++d)
            {
                values[d] = site * 100 + d + i;
            }
            auto const got = all_to_all(hpx::launch::sync, comms,
                std::move(values), this_site_arg(site), generation_arg(gen));
            HPX_TEST_EQ(got.size(), std::size_t(num_sites));
            for (u32 s = 0; s != got.size(); ++s)
            {
                HPX_TEST_EQ_MSG(got[s], u32(s * 100 + site + i),
                    ctx + " all_to_all site=" + std::to_string(site));
            }
        }

        // scans follow the actual site order, not the tree order
        {
            ++gen;
            u32 const got = inclusive_scan(hpx::launch::sync, comms,
                u32(site + 1 + i), std::plus<u32>{}, this_site_arg(site),
                generation_arg(gen));
            u32 expected = 0;
            for (u32 j = 0; j <= site; ++j)
            {
                expected += j + 1 + i;
            }
            HPX_TEST_EQ(got, expected);
        }
        {
            ++gen;
            u32 const got = exclusive_scan(hpx::launch::sync, comms,
                u32(site + 1 + i), u32(0), std::plus<u32>{},
                this_site_arg(site), generation_arg(gen));
            u32 expected = 0;
            for (u32 j = 0; j < site; ++j)
            {
                expected += j + 1 + i;
            }
            HPX_TEST_EQ(got, expected);
        }

        // barrier
        {
            ++gen;
            barrier(hpx::launch::sync, comms, this_site_arg(site),
                generation_arg(gen));
        }
    }
}

void test_roots(std::uint32_t const num_sites, std::size_t const arity,
    std::size_t const threshold)
{
    for (std::uint32_t root = 0; root != num_sites; ++root)
    {
        std::string const basename = "/test/rotating_root/" +
            std::to_string(num_sites) + "/" + std::to_string(arity) + "/" +
            std::to_string(threshold) + "/" + std::to_string(root) + "/";

        std::vector<hpx::future<void>> sites;
        sites.reserve(num_sites);
        for (std::uint32_t site = 0; site != num_sites; ++site)
        {
            sites.push_back(hpx::async([=]() {
                run_site(basename, num_sites, site, root, arity, threshold);
            }));
        }
        hpx::wait_all(sites);
        for (auto& f : sites)
        {
            f.get();
        }
    }
}

// Only the root site may call the root-side operations.
void test_root_caller_rejected()
{
    constexpr std::uint32_t num_sites = 5;
    constexpr std::uint32_t root = 3;

    std::vector<hpx::future<void>> sites;
    for (std::uint32_t site = 0; site != num_sites; ++site)
    {
        sites.push_back(hpx::async([=]() {
            auto const comms = create_hierarchical_communicator(
                "/test/rotating_root/rejected/", num_sites_arg(num_sites),
                this_site_arg(site), arity_arg(2), generation_arg(),
                root_site_arg(root), flat_fallback_threshold_arg(0));

            // site 0 is no longer the root, so it must not act as one ...
            if (site == 0)
            {
                bool rejected = false;
                try
                {
                    broadcast_to(hpx::launch::sync, comms, std::uint32_t(1),
                        this_site_arg(site), generation_arg(1));
                }
                catch (hpx::exception const&)
                {
                    rejected = true;
                }
                HPX_TEST(rejected);
            }

            // ... and the root must not call the non-root side.
            if (site == root)
            {
                bool rejected = false;
                try
                {
                    broadcast_from<std::uint32_t>(
                        comms, this_site_arg(site), generation_arg(1))
                        .get();
                }
                catch (hpx::exception const&)
                {
                    rejected = true;
                }
                HPX_TEST(rejected);
            }
        }));
    }
    hpx::wait_all(sites);
    for (auto& f : sites)
    {
        f.get();
    }

    // A root outside the participating sites is rejected.
    bool rejected = false;
    try
    {
        create_hierarchical_communicator("/test/rotating_root/out_of_range/",
            num_sites_arg(num_sites), this_site_arg(0), arity_arg(2),
            generation_arg(), root_site_arg(num_sites));
    }
    catch (hpx::exception const&)
    {
        rejected = true;
    }
    HPX_TEST(rejected);
}

int hpx_main()
{
    // All sites are simulated by tasks of this one process.
    if (hpx::get_locality_id() == 0)
    {
        for (std::uint32_t num_sites : {2u, 3u, 5u, 8u, 11u})
        {
            // a real tree of arity 2 and 3, then the flat fallback
            test_roots(num_sites, 2, 0);
            test_roots(num_sites, 3, 0);
            test_roots(num_sites, 2, 16);
        }

        test_root_caller_rejected();
    }

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
