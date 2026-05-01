#pragma once

#include <vector>
#include <optional>

// struct Table<T> {
//     size_t page_size;
//     size_t initial_page_id;
//     size_t nrows;
//     size_t npages;
//     size_t nsubpages;
//     bool compressed;
//     T *data;
// };

template <typename T>
class Table {
public:
    Table(
        size_t page_size,
        size_t initial_page_id,
        size_t nrows,
        size_t npages,
        size_t nsubpages,
        bool compressed) : page_size_(page_size),
                           initial_page_id_(initial_page_id),
                           nrows_(nrows),
                           npages_(npages),
                           nsubpages_(nsubpages),
                           compressed_(compressed)
    {

    }


    size_t get_initial_page_id() {
        return initial_page_id_;
    }

    size_t get_npages() {
        if (nsubpages_) {
            return npages_;
        } else {
            /* tables with no subpage. */
            size_t row_size = get_rec_size();
            size_t nrows_per_page = page_size_ / row_size;
            std::cout << "row_size:" << row_size << ", nrows_per_page:" << nrows_per_page;
            std::cout << ", npages:" << (nrows_ + nrows_per_page - 1) / nrows_per_page << std::endl;
            return (nrows_ + nrows_per_page - 1) / nrows_per_page;
        }
    }

    size_t get_nsubpages() {
        return nsubpages_;
    }

    size_t get_nrows() {
        return nrows_;
    }

    size_t get_rec_size() {
        return sizeof(T);
    }

    std::vector<size_t> generate_page_ids() {
        std::vector<size_t> page_ids;
        auto npages = get_npages();
        for (size_t i = 0; i < npages; i++) {
            page_ids.push_back(initial_page_id_ + i);
        }
        return page_ids;
    }

    std::optional<Table<size_t>> compressed_page_sizes_table() {
#if 1
        size_t npages = get_npages();
        return Table<size_t>(
            page_size_,
            initial_page_id_ + npages,
            npages,
            0,
            0,
            false);
#else
        if (compressed_) {
            size_t npages = get_npages();
            // The tail page is the meta page that includes the compressed page sizes
            // let nrows_of_comp_sizes_per_page = self.page_size / mem::size_of::<usize>();
            // let comp_page_sizes_npages = 
            //     if npages % nrows_of_comp_sizes_per_page == 0 {
            //         npages / nrows_of_comp_sizes_per_page
            //     } else {
            //         npages / nrows_of_comp_sizes_per_page + 1
            //     };
            return Table<size_t>(
                page_size_,
                initial_page_id_ + npages,
                npages,
                0,
                0,
                false);
        } else {
            //None
            return std::nullopt;
        }
#endif
    }

    std::optional<Table<size_t>> compressed_sub_page_sizes_table() {
#if 1
        size_t npages = get_npages();
        size_t nrows = get_nsubpages();
        size_t nrows_of_comp_sizes_per_page = page_size_ / sizeof(size_t);
        size_t comp_page_sizes_npages;
        if (npages % nrows_of_comp_sizes_per_page == 0) {
            comp_page_sizes_npages = npages / nrows_of_comp_sizes_per_page;
        } else {
            comp_page_sizes_npages  = npages / nrows_of_comp_sizes_per_page + 1;
        };
        return Table<size_t>(
            page_size_,
            initial_page_id_ + npages + comp_page_sizes_npages,
            nrows,
            0,
            0,
            false);
#else
        if (compressed_) {
            // After the compressed page sizes table, there is the compressed sub page sizes table
            // npages is the row size of the compressed page sizes table
            size_t npages = get_npages();
            size_t nrows = get_nsubpages();
            size_t nrows_of_comp_sizes_per_page = page_size_ / sizeof(size_t);
            size_t comp_page_sizes_npages;
            if (npages % nrows_of_comp_sizes_per_page == 0) {
                comp_page_sizes_npages = npages / nrows_of_comp_sizes_per_page;
            } else {
                comp_page_sizes_npages  = npages / nrows_of_comp_sizes_per_page + 1;
            };
            return Table<size_t>(
                page_size_,
                initial_page_id_ + npages + comp_page_sizes_npages,
                nrows,
                0,
                0,
                false);
        } else {
            //None
            return std::nullopt;
        }
#endif
    }

private:
    size_t page_size_;
    size_t initial_page_id_;
    size_t nrows_;
    size_t npages_;
    size_t nsubpages_;
    bool compressed_;

};

class TPCHTable {
public:
    TPCHTable(
        size_t page_size,
        size_t initial_page_id,
        size_t npages,
        size_t nrows,
        bool compressed) : page_size_(page_size),
                           initial_page_id_(initial_page_id),
                           npages_(npages),
                           nrows_(nrows),
                           compressed_(compressed)
    {

    }


    size_t get_initial_page_id() {
        return initial_page_id_;
    }

    size_t get_npages() {
        return npages_;
    }

    size_t get_nrows() {
        return nrows_;
    }

    std::vector<size_t> generate_page_ids() {
        std::vector<size_t> page_ids;
        auto npages = get_npages();
        for (size_t i = 0; i < npages; i++) {
            page_ids.push_back(initial_page_id_ + i);
        }
        return page_ids;
    }

private:
    size_t page_size_;
    size_t initial_page_id_;
    size_t npages_;
    size_t nrows_;
    bool compressed_;

};
