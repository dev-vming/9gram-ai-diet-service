import { UserService } from 'src/user/user.service';
import { Body, Controller,Param, Post, Res } from "@nestjs/common";
import { ImageService } from "./image.service";
import { ApiParam, ApiProperty, ApiTags } from "@nestjs/swagger";
import { CreateImageDto } from './dtos/CreateImage.dto';

@ApiTags('image')
@Controller("image")
export class ImageController {
    constructor(
        private imageService: ImageService,
    ) { }

    // presigned url for uploading food image
    @Post('/presigned-url/food/:filename')
    @ApiProperty( {description: "get presigned url for put"} )
    @ApiParam({ name: 'filename', type: 'string', description: 'Name of the file to upload' })
    async handleGetPresignedUrlForFood(@Param('filename') fileName: string) {
        const presinedUrl = await this.imageService.getPresignedUrlForPut(`food/${fileName}`);
        return presinedUrl;
    }

    // presigned url for uploading profile image
    @Post('/presigned-url/profile/:filename')
    @ApiProperty( {description: "get presigned url for put"} )
    @ApiParam({ name: 'filename', type: 'string', description: 'Name of the file to upload' })
    async handleGetPresignedUrlForProfile(@Param('filename') fileName: string) {
        return this.imageService.getPresignedUrlForPut(`profile/${fileName}`);
    }
}