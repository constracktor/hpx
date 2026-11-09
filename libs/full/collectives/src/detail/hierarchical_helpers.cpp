//  Copyright (c) 2026 Anshuman Agrawal
//  Copyright (c) 2026 Alexander Strack
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/assert.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/detail/hierarchical_helpers.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>

#include <cstddef>
#include <exception>

namespace hpx::collectives::detail {

    std::exception_ptr validate_hierarchical_communicator(
        hierarchical_communicator const& communicators,
        this_site_arg const this_site, char const* operation)
    {
        if (!communicators.valid())
        {
            return HPX_GET_EXCEPTION(hpx::error::invalid_status, operation,
                "the hierarchical communicator is not valid");
        }

        auto const [num_sites_val, communicator_site] =
            communicators.get_info();

        // An out-of-range site belongs to no sub-communicator of the tree and
        // would hang the collective.
        if (this_site >= num_sites_val)
        {
            return HPX_GET_EXCEPTION(hpx::error::bad_parameter, operation,
                "this_site must be smaller than the number of participating "
                "sites");
        }

        if (this_site != communicator_site)
        {
            return HPX_GET_EXCEPTION(hpx::error::bad_parameter, operation,
                "this_site must match the site used to create the "
                "hierarchical communicator");
        }

        return std::exception_ptr();
    }

    std::exception_ptr validate_hierarchical_root_caller(
        hierarchical_communicator const& communicators,
        this_site_arg const this_site, char const* operation,
        char const* root_operation)
    {
        if (this_site != communicators.get_root_site())
        {
            return HPX_GET_EXCEPTION(hpx::error::bad_parameter, operation,
                hpx::util::format("only the root site may call {} on a "
                                  "hierarchical communicator",
                    root_operation));
        }

        return std::exception_ptr();
    }

    std::exception_ptr validate_hierarchical_non_root_caller(
        hierarchical_communicator const& communicators,
        this_site_arg const this_site, char const* operation,
        char const* root_operation)
    {
        if (this_site == communicators.get_root_site())
        {
            return HPX_GET_EXCEPTION(hpx::error::bad_parameter, operation,
                hpx::util::format(
                    "the root site must call {} on a hierarchical communicator",
                    root_operation));
        }

        return std::exception_ptr();
    }

    std::exception_ptr validate_hierarchical_root_site(
        hierarchical_communicator const& communicators,
        root_site_arg const root_site, char const* operation,
        char const* operation_name)
    {
        // 0 is accepted so callers that never named a root keep working; it
        // can't be told from an explicit root_site_arg(0), which is therefore
        // accepted on a communicator rooted elsewhere as well. Anything else
        // must be the root the tree was built with.
        if (!root_site.is_default() &&
            root_site != communicators.get_root_site())
        {
            return HPX_GET_EXCEPTION(hpx::error::bad_parameter, operation,
                hpx::util::format(
                    "hierarchical {} must use the root site the communicator "
                    "was created with ({})",
                    operation_name,
                    static_cast<std::size_t>(communicators.get_root_site())));
        }

        return std::exception_ptr();
    }

}    // namespace hpx::collectives::detail

#endif
