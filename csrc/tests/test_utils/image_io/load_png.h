// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once

#include <string>
#include <memory>
#include "tests/test_utils/pixel.h"
#include "tests/test_utils/image_view.h"

namespace test{

struct LibpngData;
struct PngBufferReaderState;
struct FileInfo;
class png_loader {
public:

	png_loader( const char* filename );
	png_loader( const std::string& filename );
	// png_loader( const dlib::file& f );
	png_loader( const unsigned char* image_buffer, size_t buffer_size );
	~png_loader();

	bool is_gray() const;
	bool is_graya() const;
	bool is_rgb() const;
	bool is_rgba() const;

	unsigned int bit_depth () const { return bit_depth_; }

	template<typename T>
	void get_image( T& t_) const{
		typedef typename image_traits<T>::pixel_type pixel_type;
		image_view<T> t(t_);
		t.set_size( height_, width_ );


		if (is_gray() && bit_depth_ == 8)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const unsigned char* v = get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					unsigned char p = v[m];
					assign_pixel( t[n][m], p );
				}
			}
		}
		else if (is_gray() && bit_depth_ == 16)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const test::uint16* v = (test::uint16*)get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					test::uint16 p = v[m];
					assign_pixel( t[n][m], p );
				}
			}
		}
		else if (is_graya() && bit_depth_ == 8)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const unsigned char* v = get_row( n );
				for ( unsigned m = 0; m < width_; m++ )
				{
					unsigned char p = v[m*2];
					if (!pixel_traits<pixel_type>::has_alpha)
					{
						assign_pixel( t[n][m], p );
					}
					else
					{
						unsigned char pa = v[m*2+1];
						rgb_alpha_pixel pix;
						assign_pixel(pix, p);
						assign_pixel(pix.alpha, pa);
						assign_pixel(t[n][m], pix);
					}
				}
			}
		}
		else if (is_graya() && bit_depth_ == 16)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const test::uint16* v = (test::uint16*)get_row( n );
				for ( unsigned m = 0; m < width_; m++ )
				{
					test::uint16 p = v[m*2];
					if (!pixel_traits<pixel_type>::has_alpha)
					{
						assign_pixel( t[n][m], p );
					}
					else
					{
						test::uint16 pa = v[m*2+1];
						rgb_alpha_pixel pix;
						assign_pixel(pix, p);
						assign_pixel(pix.alpha, pa);
						assign_pixel(t[n][m], pix);
					}
				}
			}
		}
		else if (is_rgb() && bit_depth_ == 8)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const unsigned char* v = get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					rgb_pixel p;
					p.red = v[m*3];
					p.green = v[m*3+1];
					p.blue = v[m*3+2];
					assign_pixel( t[n][m], p );
				}
			}
		}
		else if (is_rgb() && bit_depth_ == 16)
		{
			for ( unsigned n = 0; n < height_;n++ )
			{
				const uint16* v = (uint16*)get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					rgb_pixel p;
					p.red   = static_cast<uint8>(v[m*3]);
					p.green = static_cast<uint8>(v[m*3+1]);
					p.blue  = static_cast<uint8>(v[m*3+2]);
					assign_pixel( t[n][m], p );
				}
			}
		}
		else if (is_rgba() && bit_depth_ == 8)
		{
			if (!pixel_traits<pixel_type>::has_alpha)
				assign_all_pixels(t,0);

			for ( unsigned n = 0; n < height_;n++ )
			{
				const unsigned char* v = get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					rgb_alpha_pixel p;
					p.red = v[m*4];
					p.green = v[m*4+1];
					p.blue = v[m*4+2];
					p.alpha = v[m*4+3];
					assign_pixel( t[n][m], p );
				}
			}
		}
		else if (is_rgba() && bit_depth_ == 16)
		{
			if (!pixel_traits<pixel_type>::has_alpha)
				assign_all_pixels(t,0);

			for ( unsigned n = 0; n < height_;n++ )
			{
				const uint16* v = (uint16*)get_row( n );
				for ( unsigned m = 0; m < width_;m++ )
				{
					rgb_alpha_pixel p;
					p.red   = static_cast<uint8>(v[m*4]);
					p.green = static_cast<uint8>(v[m*4+1]);
					p.blue  = static_cast<uint8>(v[m*4+2]);
					p.alpha = static_cast<uint8>(v[m*4+3]);
					assign_pixel( t[n][m], p );
				}
			}
		}
	}

private:
	const unsigned char* get_row( unsigned i ) const;
	std::unique_ptr<FileInfo> check_file( const char* filename );
	void read_image( std::unique_ptr<FileInfo> file_info );
	unsigned height_, width_;
	unsigned bit_depth_;
	int color_type_;
	std::unique_ptr<LibpngData> ld_;
	std::unique_ptr<PngBufferReaderState> buffer_reader_state_;
};

// ----------------------------------------------------------------------------------------

template <
		typename image_type
>
void load_png (
		image_type& image,
		const std::string& file_name
)
{
	png_loader(file_name).get_image(image);
}

template <
		typename image_type
>
void load_png (
		image_type& image,
		const unsigned char* image_buffer,
		size_t buffer_size
)
{
	png_loader(image_buffer, buffer_size).get_image(image);
}

template <
		typename image_type
>
void load_png (
		image_type& image,
		const char* image_buffer,
		size_t buffer_size
)
{
	png_loader(reinterpret_cast<const unsigned char*>(image_buffer), buffer_size).get_image(image);
}


// ----------------------------------------------------------------------------------------

} // namespace test